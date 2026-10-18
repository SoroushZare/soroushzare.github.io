"""Effectors: every backend clips and scales the command, stops safely on close, and the simulated arm
follows a commanded end-effector velocity without blowing up at singular poses."""
import json
import socket
import sys
import types

import numpy as np
import pytest

from neural_cursor.effectors import (Effector, NullEffector, PlanarArm2Link, ROS2Twist,
                                     ScreenCursor, UDPVelocity)

DT = 0.01   # 100 Hz integration step for the arm tests


# ---------------------------------------------------------------------------------------- base
def test_base_requires_apply():
    with pytest.raises(NotImplementedError):
        Effector(dim=2, max_speed=1.0).send([0.1, 0.2], 0.1)


def test_null_effector_clips_and_rejects_bad_shape():
    eff = NullEffector()
    eff.send([3.0, -7.0], 0.1)
    assert eff.state()["v"] == [1.0, -1.0]
    with pytest.raises(ValueError):
        eff.send([0.1, 0.2, 0.3], 0.1)
    with pytest.raises(ValueError):
        eff.send([0.1, 0.2], -0.1)
    eff.close()
    with pytest.raises(RuntimeError):
        eff.send([0.0, 0.0], 0.1)


def test_non_finite_command_means_stop():
    cur = ScreenCursor(1000, 800, gain_px_s=500, backend="log")
    start = cur.pos.copy()
    cur.send([np.nan, np.inf], 0.1)
    np.testing.assert_allclose(cur.pos, start)
    assert cur.state()["v"] == [0.0, 0.0]
    cur.send([np.nan, 1.0], 0.1)                      # one bad axis stops every axis
    np.testing.assert_allclose(cur.pos, start)
    assert cur.state()["v"] == [0.0, 0.0]


def test_rejected_send_changes_no_state():
    eff = NullEffector()
    eff.send([0.5, 0.5], 0.1)
    with pytest.raises(ValueError):
        eff.send([0.9, -0.9], np.nan)
    assert eff.state()["v"] == [0.5, 0.5]


def test_screen_cursor_reports_its_position_in_task_units():
    cur = ScreenCursor(1001, 801, gain_px_s=100, backend="log")
    np.testing.assert_allclose(cur.task_position(), [0.0, 0.0])
    cur.send([1.0, 1.0], 1.0)                         # 100 px right, 100 px up
    np.testing.assert_allclose(cur.task_position(), [100 / 500, 100 / 400])
    assert NullEffector().task_position() is None


# ---------------------------------------------------------------------------------------- screen
def test_screen_cursor_integrates_in_pixels_with_y_up():
    cur = ScreenCursor(1000, 800, gain_px_s=400, backend="log")
    assert cur.backend == "log"
    start = cur.pos.copy()
    cur.send([0.5, 0.25], 0.1)                        # 0.5·400·0.1 = 20 px right, 10 px up
    np.testing.assert_allclose(cur.pos - start, [20.0, -10.0])
    # Sub-pixel steps accumulate instead of being rounded away.
    for _ in range(10):
        cur.send([0.01, 0.0], 0.1)                    # 0.4 px each
    np.testing.assert_allclose(cur.pos[0] - start[0], 24.0)


def test_screen_cursor_clips_command_and_clamps_to_screen():
    cur = ScreenCursor(1000, 800, gain_px_s=400, backend="log")
    start = cur.pos.copy()
    cur.send([5.0, 0.0], 0.1)                         # clipped to 1 → 40 px, not 200
    np.testing.assert_allclose(cur.pos[0] - start[0], 40.0)
    for _ in range(100):
        cur.send([1.0, -1.0], 0.1)                    # far beyond the right / bottom edges
    x, y = cur.state()["pos"]
    assert x == 1000 - 2 and y == 800 - 2            # 1 px edge margin, 0-indexed pixels
    for _ in range(100):
        cur.send([-1.0, 1.0], 0.1)
    assert cur.state()["pos"] == [1.0, 1.0]


def test_screen_cursor_auto_uses_pyautogui_when_importable(monkeypatch):
    calls = []
    fake = types.ModuleType("pyautogui")
    fake.moveTo = lambda x, y, _pause=True: calls.append((x, y, _pause))
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    cur = ScreenCursor(1001, 801, gain_px_s=100, backend="auto")
    assert cur.backend == "pyautogui"
    cur.send([1.0, 1.0], 0.1)
    assert calls[0] == (500, 400, False)              # centred on construction, no PAUSE sleep
    assert calls[-1] == (510, 390, False)
    assert all(isinstance(c[0], int) and isinstance(c[1], int) for c in calls)


def test_screen_cursor_auto_falls_back_without_display(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyautogui", None)   # import raises ImportError
    assert ScreenCursor(640, 480, gain_px_s=100).backend == "log"


# ---------------------------------------------------------------------------------------- UDP
@pytest.fixture
def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2.0)
    yield sock
    sock.close()


def _recv(sock):
    return json.loads(sock.recv(65536).decode("utf-8"))


def test_udp_sends_clipped_scaled_json(receiver):
    port = receiver.getsockname()[1]
    eff = UDPVelocity("127.0.0.1", port, scale=0.5, clock=lambda: 123.0)
    eff.send([0.4, -3.0], 0.1)
    eff.send([0.0, 1.0], 0.1)
    m0, m1 = _recv(receiver), _recv(receiver)
    assert m0 == {"t": 123.0, "seq": 0, "v": [0.2, -0.5]}
    assert m1["seq"] == 1 and m1["v"] == [0.0, 0.5]
    eff.close()
    stop = _recv(receiver)                            # close() always sends a final stop
    assert stop["seq"] == 2 and stop["v"] == [0.0, 0.0]
    assert eff.state()["n_errors"] == 0
    eff.close()                                       # idempotent: no second stop, no error


def test_udp_dimension_is_configurable(receiver):
    port = receiver.getsockname()[1]
    eff = UDPVelocity("127.0.0.1", port, scale=2.0, dim=3)
    eff.send([0.1, 0.2, 0.3], 0.1)
    np.testing.assert_allclose(_recv(receiver)["v"], [0.2, 0.4, 0.6])
    eff.close()


# ---------------------------------------------------------------------------------------- ROS 2
def test_ros2_missing_rclpy_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", None)
    with pytest.raises(ImportError, match="ROS 2"):
        ROS2Twist()


def _fake_ros(monkeypatch):
    """Minimal rclpy + geometry_msgs stand-ins that record what is published."""
    published, state = [], {"ok": False, "shutdown": False, "destroyed": False}

    class Vec:
        x = y = z = 0.0

    class Twist:
        def __init__(self):
            self.linear, self.angular = Vec(), Vec()

    class Pub:
        def publish(self, msg):
            published.append(((msg.linear.x, msg.linear.y, msg.linear.z),
                              (msg.angular.x, msg.angular.y, msg.angular.z)))

    class Node:
        def create_publisher(self, msg_type, topic, depth):
            assert msg_type is Twist and topic == "/cmd_vel"
            return Pub()

        def destroy_node(self):
            state["destroyed"] = True

    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: state["ok"]
    rclpy.init = lambda: state.update(ok=True)
    rclpy.shutdown = lambda: state.update(shutdown=True)
    rclpy.create_node = lambda name: Node()
    msg = types.ModuleType("geometry_msgs.msg")
    msg.Twist = Twist
    pkg = types.ModuleType("geometry_msgs")
    pkg.msg = msg
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "geometry_msgs", pkg)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", msg)
    return published, state


def test_ros2_diff_drive_mapping_and_stop_on_close(monkeypatch):
    published, state = _fake_ros(monkeypatch)
    eff = ROS2Twist(scale=0.2, angular_scale=0.8)
    eff.send([0.5, 2.0], 0.1)                         # right 0.5, up clipped to 1
    lin, ang = published[0]
    np.testing.assert_allclose(lin, [0.2, 0.0, 0.0])
    np.testing.assert_allclose(ang, [0.0, 0.0, -0.4])
    eff.close()
    assert published[-1] == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert state["destroyed"] and state["shutdown"]


def test_ros2_holonomic_mapping(monkeypatch):
    published, _ = _fake_ros(monkeypatch)
    eff = ROS2Twist(scale=0.3, mode="holonomic")
    eff.send([1.0, -0.5], 0.1)
    lin, ang = published[0]
    np.testing.assert_allclose(lin, [-0.15, -0.3, 0.0])
    np.testing.assert_allclose(ang, [0.0, 0.0, 0.0])
    eff.close()


# ---------------------------------------------------------------------------------------- arm
@pytest.mark.parametrize("v", [(1, 0), (-1, 0), (0, 1), (0, -1), (0.6, 0.8), (-0.7, -0.7)])
def test_arm_moves_end_effector_in_commanded_direction(v):
    arm = PlanarArm2Link()
    p0 = arm.fk()
    for _ in range(20):                               # 0.2 s at 0.25 m/s full speed
        arm.send(v, DT)
    d = arm.fk() - p0
    u = np.asarray(v, float) / np.linalg.norm(v)
    assert np.dot(d, u) / np.linalg.norm(d) > 0.99    # direction
    # Speed is |v| × max_speed, shrunk by the DLS gain σ²/(σ² + λ²) ≥ 0.92 along the weakest
    # Jacobian direction at the start pose (σ_min ≈ 0.175 m, λ = 0.05 m): damping trades a few
    # percent of tracking for bounded joint speed, so the band is [0.85, 1].
    ratio = np.linalg.norm(d) / (0.2 * arm.max_speed * np.linalg.norm(v))
    assert 0.85 <= ratio <= 1.0 + 1e-9


def test_arm_singular_pose_does_not_blow_up():
    # Fully stretched (q2 = 0): det J = 0, where a plain inverse Jacobian is undefined.
    arm = PlanarArm2Link(q0=(np.pi / 2, 0.0), workspace=None)
    assert abs(np.linalg.det(arm.jacobian())) < 1e-12
    bound = arm.max_speed * np.sqrt(2) / (2 * arm.damping)   # DLS: |qdot| ≤ |xdot| / (2λ)
    for v in [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (-1.0, 0.0)]:
        for _ in range(50):
            arm.send(v, DT)
            assert np.all(np.isfinite(arm.q)) and np.all(np.isfinite(arm.qdot))
            assert np.linalg.norm(arm.qdot) <= bound + 1e-9
            assert np.max(np.abs(arm.qdot)) <= arm.max_joint_speed + 1e-9
    # Folded-back is the other singularity family; the elbow limit keeps it out of reach and the arm
    # still moves sensibly from its limit.
    arm = PlanarArm2Link(q0=(0.0, 2.8))
    for _ in range(100):
        arm.send((0.0, -1.0), DT)
        assert np.all(np.isfinite(arm.q))


def test_arm_joint_limits_and_workspace_are_enforced():
    arm = PlanarArm2Link()
    xmin, xmax, ymin, ymax = arm.workspace
    rng = np.random.default_rng(0)
    for _ in range(3000):                             # 30 s of random full-speed pushing
        arm.send(rng.uniform(-1.5, 1.5, 2), DT)
        assert np.all(arm.q >= arm.q_lo - 1e-12) and np.all(arm.q <= arm.q_hi + 1e-12)
        x, y = arm.fk()
        assert xmin - 1e-3 <= x <= xmax + 1e-3 and ymin - 1e-3 <= y <= ymax + 1e-3
    for _ in range(500):                              # push into the floor: it slides, never crosses
        arm.send((0.3, -1.0), DT)
    assert arm.fk()[1] >= ymin - 1e-3
    st = arm.state()
    assert set(st) >= {"pos", "q", "qdot", "t", "v"}


@pytest.mark.parametrize("dt", [0.1, 0.5, 1.0])
def test_arm_long_steps_stay_in_the_workspace(dt):
    """A scheduler stall hands the arm one long dt; it is integrated in substeps, so the first-order
    workspace clamp still holds (a single Euler step overran the floor by up to 0.3 m)."""
    xmin, xmax, ymin, ymax = PlanarArm2Link().workspace
    arm = PlanarArm2Link(q0=(0.442, 2.8))                       # the reviewer's worst case
    arm.send((1.0, 1.0), dt)
    assert arm.fk()[1] >= ymin - 1e-3
    rng = np.random.default_rng(1)
    for _ in range(40):
        arm = PlanarArm2Link()
        for _ in range(5):
            arm.send(rng.uniform(-1, 1, 2), DT)
        arm.send(rng.uniform(-1.5, 1.5, 2), dt)
        x, y = arm.fk()
        assert xmin - 1e-3 <= x <= xmax + 1e-3 and ymin - 1e-3 <= y <= ymax + 1e-3
        assert np.hypot(x, y) <= arm.r_max + 1e-3


def test_arm_rejects_a_start_pose_outside_the_workspace():
    with pytest.raises(ValueError, match="outside the workspace"):
        PlanarArm2Link(q0=(-0.5, 0.3))                          # end effector below the floor
    PlanarArm2Link(q0=(-0.5, 0.3), workspace=None)              # fine without a box


def test_arm_zero_dt_is_a_no_op():
    arm = PlanarArm2Link()
    q0 = arm.q.copy()
    arm.send((1.0, 1.0), 0.0)
    np.testing.assert_array_equal(arm.q, q0)
