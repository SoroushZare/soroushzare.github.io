"""Effectors: where a decoded velocity leaves the software and moves something.

Every effector takes the decoder's command ``v`` in [−1, 1]^k (unit = full speed along that axis) and a
time step ``dt`` in seconds.  The contract is deliberately the same for a screen cursor, a UDP robot,
a ROS 2 base and a simulated arm, so the real-time loop never knows what it is driving and an effector
can be swapped without touching the decoder.

Safety rules shared by all effectors (enforced in ``Effector._command``):

* ``v`` is clipped to [−1, 1] and multiplied by an explicit, per-effector ``max_speed`` stated in the
  effector's own units (px/s, m/s, ...).  A decoder can therefore never exceed the configured speed,
  whatever its output scale.
* A command with any non-finite entry (NaN/inf, e.g. from a decoder fed a flat-lined window) becomes
  the zero vector — all axes, not just the bad one: for anything that moves in the world "stop" is
  the only safe interpretation of garbage, and half a garbage command is still garbage.
* ``dt`` is validated before anything else, so a rejected call changes no state.
* ``close()`` brings the effector to rest (a final zero command for velocity-controlled devices),
  because a velocity robot otherwise keeps executing the last command it heard.

Pure-EEG rule: effectors receive only decoder (EEG-derived) commands.  Nothing here reads an auditor
stream (eye tracker, EMG); those only ever reach ``purity.py``.
"""
from __future__ import annotations

import importlib
import json
import logging
import socket
import time
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


class Effector:
    """Base class.  Subclasses implement ``_apply(u, dt)`` with ``u`` already clipped and scaled.

    ``dim`` is the command dimension k; ``max_speed`` is the speed (in ``units``) produced by a
    command of 1 along an axis.
    """

    units: str = "a.u./s"

    def __init__(self, dim: int, max_speed: float):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if not np.isfinite(max_speed) or max_speed <= 0:
            raise ValueError(f"max_speed must be a positive finite number, got {max_speed}")
        self.dim = int(dim)
        self.max_speed = float(max_speed)
        self._last_v = np.zeros(self.dim)
        self._closed = False

    # -- public contract ------------------------------------------------------------------------
    def send(self, v: np.ndarray, dt: float) -> None:
        """Apply command ``v`` (in [−1, 1]^dim; clipped if not) for ``dt`` seconds."""
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0:
            raise ValueError(f"dt must be finite and >= 0, got {dt}")
        u = self._command(v)
        self._apply(u * self.max_speed, dt)

    def state(self) -> dict:
        """Loggable state; subclasses extend it.  ``v`` is the last clipped (unscaled) command."""
        return {"v": self._last_v.tolist(), "max_speed": self.max_speed, "units": self.units}

    def task_position(self) -> Optional[np.ndarray]:
        """Where the person sees the cursor, in normalised task units ([−1, 1]², +y up), or None when
        this effector draws no cursor.  The real-time loop logs this position (not its internal model)
        so visible-cursor labels refer to what was actually on screen."""
        return None

    def close(self) -> None:
        """Bring the device to rest and release resources.  Idempotent."""
        if self._closed:
            return
        self._stop()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- subclass hooks -------------------------------------------------------------------------
    def _apply(self, u: np.ndarray, dt: float) -> None:
        raise NotImplementedError

    def _stop(self) -> None:
        """Default: a position-integrating effector is at rest as soon as commands stop."""

    # -- shared sanitation ----------------------------------------------------------------------
    def _command(self, v) -> np.ndarray:
        """Validate shape, zero non-finite entries, clip to [−1, 1]; remember it for ``state``."""
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.shape != (self.dim,):
            raise ValueError(f"{type(self).__name__} expects a command of length {self.dim}, "
                             f"got shape {v.shape}")
        if not np.isfinite(v).all():
            log.warning("%s: non-finite command %s replaced by 0 (stop)", type(self).__name__, v)
            v = np.zeros(self.dim)
        self._last_v = np.clip(v, -1.0, 1.0)
        return self._last_v


class NullEffector(Effector):
    """Discards commands.  Used for offline replay and blank-feedback probes where nothing may move."""

    def __init__(self, dim: int = 2, max_speed: float = 1.0):
        super().__init__(dim, max_speed)

    def _apply(self, u: np.ndarray, dt: float) -> None:
        return None


class ScreenCursor(Effector):
    """OS mouse cursor driven by velocity.

    Position is integrated in floating-point pixels and only rounded when handed to the OS, so slow
    commands (< 1 px per step) still accumulate instead of being truncated to zero forever.

    Coordinates follow the task convention (+x right, +y up); the screen's y axis points down, so the
    y component is negated when integrating.  The cursor starts at the screen centre.

    Backends:
      * ``"auto"`` - pyautogui if it imports (it needs a display), else the log-only backend.
      * ``"log"``  - integrate and clamp only; nothing is moved.  For headless CI and replay.

    Clamp: the cursor is kept ``edge_margin_px`` inside the screen.  The default of 1 px keeps a
    drifting decoder off the exact corner pixels, which are pyautogui's fail-safe trigger; a human
    slamming the physical mouse into a corner still trips it, which is the intended emergency stop.
    """

    units = "px/s"

    def __init__(self, width_px: int, height_px: int, gain_px_s: float, backend: str = "auto",
                 edge_margin_px: int = 1):
        super().__init__(dim=2, max_speed=gain_px_s)
        if width_px <= 2 * edge_margin_px or height_px <= 2 * edge_margin_px:
            raise ValueError("screen must be larger than twice the edge margin")
        if backend not in ("auto", "log"):
            raise ValueError(f"backend must be 'auto' or 'log', got {backend!r}")
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self._lo = np.array([edge_margin_px, edge_margin_px], float)
        self._hi = np.array([width_px - 1 - edge_margin_px, height_px - 1 - edge_margin_px], float)
        self.pos = np.array([(width_px - 1) / 2.0, (height_px - 1) / 2.0])
        self._t = 0.0
        self._gui = self._load_pyautogui() if backend == "auto" else None
        self.backend = "pyautogui" if self._gui is not None else "log"
        if self._gui is not None:
            self._move_os()

    @staticmethod
    def _load_pyautogui():
        """Import pyautogui or return None.

        Any exception, not only ImportError, means "no usable GUI": on a headless Linux box pyautogui
        fails at import time with KeyError('DISPLAY') or an Xlib error.
        """
        try:
            return importlib.import_module("pyautogui")
        except Exception as exc:                     # noqa: BLE001 - see docstring
            log.info("pyautogui unavailable (%s: %s); ScreenCursor uses the log-only backend",
                     type(exc).__name__, exc)
            return None

    def _move_os(self) -> None:
        # _pause=False: pyautogui otherwise sleeps PAUSE (0.1 s) after every call, i.e. one whole
        # control period of added latency.
        x, y = np.rint(self.pos).astype(int)
        self._gui.moveTo(int(x), int(y), _pause=False)

    def _apply(self, u: np.ndarray, dt: float) -> None:
        step = np.array([u[0], -u[1]]) * dt
        self.pos = np.clip(self.pos + step, self._lo, self._hi)
        self._t += dt
        if self._gui is not None:
            self._move_os()

    def state(self) -> dict:
        return {**super().state(), "pos": self.pos.tolist(), "t": self._t, "backend": self.backend}

    def task_position(self) -> np.ndarray:
        """Pixel position → task units: the screen centre is (0, 0), the edges ±1, +y up."""
        half = np.array([(self.width_px - 1) / 2.0, (self.height_px - 1) / 2.0])
        rel = (self.pos - half) / half
        return np.array([rel[0], -rel[1]])


class UDPVelocity(Effector):
    """Velocity datagrams for any robot that can read UDP.

    One JSON object per ``send``: ``{"t": <unix time>, "seq": <int>, "v": [v_1, ..., v_k]}`` with
    ``v`` = clipped command × ``scale``.  ``t`` is wall-clock (``time.time``) rather than monotonic
    because the receiver is usually another machine and only wall-clock time is comparable across
    hosts (given NTP).  ``seq`` lets the receiver detect loss and reordering, which UDP does not
    report.  The receiver should implement its own command timeout: if datagrams stop, it must stop.

    A send that fails at the OS level is counted and logged, not raised: one dropped datagram must
    not kill the control loop, and the next datagram supersedes it anyway.
    """

    units = "receiver units/s"

    def __init__(self, host: str, port: int, scale: float = 1.0, dim: int = 2,
                 clock: Callable[[], float] = time.time):
        super().__init__(dim=dim, max_speed=scale)
        self.addr = (host, int(port))
        self.clock = clock
        self.seq = 0
        self.n_errors = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _apply(self, u: np.ndarray, dt: float) -> None:
        payload = json.dumps({"t": self.clock(), "seq": self.seq, "v": [float(x) for x in u]})
        self.seq += 1
        try:
            self._sock.sendto(payload.encode("utf-8"), self.addr)
        except OSError as exc:
            self.n_errors += 1
            log.warning("UDPVelocity send to %s failed: %s", self.addr, exc)

    def _stop(self) -> None:
        self._last_v = np.zeros(self.dim)
        self._apply(np.zeros(self.dim), 0.0)
        self._sock.close()

    def state(self) -> dict:
        return {**super().state(), "seq": self.seq, "n_errors": self.n_errors}


class ROS2Twist(Effector):
    """Publishes ``geometry_msgs/Twist`` on a ROS 2 topic (default ``/cmd_vel``).

    ``rclpy`` is imported only when this class is constructed, so the rest of the package runs on
    machines without ROS; if it is missing the ImportError says what to install.

    Mapping (ROS REP-103: x forward, y left, z up), command ``v = (v_right, v_up)``:
      * ``mode="diff_drive"`` (default; most wheeled bases): up → ``linear.x`` (forward, ``scale``
        m/s), right → ``angular.z`` (negative = clockwise, ``angular_scale`` rad/s).
      * ``mode="holonomic"``: up → ``linear.x``, right → ``-linear.y``, both at ``scale`` m/s.

    ``close()`` publishes a zero Twist before tearing down the node: a base keeps the last
    ``cmd_vel`` until its own timeout, so leaving without a stop is unsafe.
    """

    units = "m/s"

    def __init__(self, topic: str = "/cmd_vel", scale: float = 0.2, angular_scale: float = 0.8,
                 mode: str = "diff_drive", node_name: str = "neural_cursor_effector",
                 qos_depth: int = 10):
        super().__init__(dim=2, max_speed=scale)
        if mode not in ("diff_drive", "holonomic"):
            raise ValueError(f"mode must be 'diff_drive' or 'holonomic', got {mode!r}")
        if scale <= 0 or angular_scale <= 0:
            raise ValueError("scale and angular_scale must be positive")
        try:
            rclpy = importlib.import_module("rclpy")
            twist_cls = importlib.import_module("geometry_msgs.msg").Twist
        except ImportError as exc:
            raise ImportError(
                "ROS2Twist needs ROS 2 Python bindings (rclpy and geometry_msgs). Install a ROS 2 "
                "distribution (e.g. Humble/Jazzy) and `source /opt/ros/<distro>/setup.bash` before "
                f"starting Python, or use UDPVelocity instead. Original error: {exc}") from exc
        self.topic = topic
        self.scale = float(scale)
        self.angular_scale = float(angular_scale)
        self.mode = mode
        self._rclpy = rclpy
        self._twist_cls = twist_cls
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._pub = self._node.create_publisher(twist_cls, topic, qos_depth)
        self.n_published = 0

    def twist_components(self, u: np.ndarray) -> dict:
        """Map a clipped command to Twist fields; exposed so the mapping is testable without ROS."""
        right, up = float(u[0]), float(u[1])
        if self.mode == "diff_drive":
            return {"linear": (up * self.scale, 0.0, 0.0),
                    "angular": (0.0, 0.0, -right * self.angular_scale)}
        return {"linear": (up * self.scale, -right * self.scale, 0.0), "angular": (0.0, 0.0, 0.0)}

    def _publish(self, u: np.ndarray) -> None:
        comp = self.twist_components(u)
        msg = self._twist_cls()
        msg.linear.x, msg.linear.y, msg.linear.z = comp["linear"]
        msg.angular.x, msg.angular.y, msg.angular.z = comp["angular"]
        self._pub.publish(msg)
        self.n_published += 1

    def _apply(self, u: np.ndarray, dt: float) -> None:
        # The linear and angular axes have different scales, so map from the clipped command rather
        # than from the single-speed product ``u`` the base class computes.
        self._publish(self._last_v)

    def _stop(self) -> None:
        self._last_v = np.zeros(2)
        self._publish(np.zeros(2))
        self._node.destroy_node()
        if self._owns_context:
            self._rclpy.shutdown()

    def state(self) -> dict:
        return {**super().state(), "topic": self.topic, "mode": self.mode,
                "n_published": self.n_published}


class PlanarArm2Link(Effector):
    """Simulated planar 2-link arm whose end effector follows the decoded velocity.

    Kinematics: base at the origin, joint angles q = (q1, q2) with q1 measured from +x and q2 relative
    to link 1.  The command is an end-effector velocity ``xdot = v × max_speed`` (m/s).  Each step:

    1. **Workspace clamp** - the next end-effector point ``p + xdot·dt`` is projected into the allowed
       region (the axis-aligned ``workspace`` box intersected with the reachable annulus implied by the
       elbow limits), and ``xdot`` is recomputed from the projected point.  Pushing against a wall
       therefore slides along it instead of fighting it through the inverse kinematics.
    2. **Damped least squares** - ``qdot = Jᵀ (J Jᵀ + λ² I)⁻¹ xdot`` with λ = ``damping`` (0.05 m).
       Plain J⁻¹ explodes as det J → 0 (arm straight, q2 = 0, or folded); DLS bounds
       ``|qdot| ≤ |xdot| / (2λ)`` everywhere, trading a small tracking error near singularities for
       never commanding a violent joint motion.
    3. **Joint speed limit** - ``qdot`` is scaled down uniformly (direction preserved) if any joint
       exceeds ``max_joint_speed``.
    4. **Joint limits** - ``q + qdot·dt`` is clipped to ``joint_limits``.  The default elbow range
       [0, 2.8] rad keeps the arm in one elbow-up branch, so it can never flip through the straight
       singularity into the mirror configuration.
    5. **Workspace check** - a joint step whose forward kinematics land more than ``WORKSPACE_TOL_M``
       outside the allowed region is rejected (the arm holds still for that substep).

    Steps 1–5 are first-order (Euler) and only accurate for short steps, so a ``dt`` longer than
    ``MAX_SUBSTEP_S`` is integrated in equal substeps of at most that length: after a scheduler stall
    one long step would otherwise jump far past the workspace walls the clamp is meant to enforce.
    The start pose ``q0`` must lie inside the joint limits' clip and the allowed region.
    """

    MAX_SUBSTEP_S = 0.01        # longest single Euler step (100 Hz); the clamp's error is O(step²)
    WORKSPACE_TOL_M = 1e-3      # 1 mm: tolerance for the first-order step overshooting a wall

    units = "m/s"

    def __init__(self, l1: float = 0.3, l2: float = 0.25, max_speed: float = 0.25,
                 damping: float = 0.05, max_joint_speed: float = 3.0,
                 joint_limits: tuple = ((-np.pi, np.pi), (0.0, 2.8)),
                 workspace: Optional[tuple] = (-0.55, 0.55, 0.0, 0.55),
                 q0: tuple = (np.pi / 4, np.pi / 2)):
        super().__init__(dim=2, max_speed=max_speed)
        if l1 <= 0 or l2 <= 0:
            raise ValueError("link lengths must be positive")
        self.l1, self.l2 = float(l1), float(l2)
        self.damping = float(damping)
        self.max_joint_speed = float(max_joint_speed)
        self.q_lo = np.array([joint_limits[0][0], joint_limits[1][0]], float)
        self.q_hi = np.array([joint_limits[0][1], joint_limits[1][1]], float)
        self.workspace = None if workspace is None else tuple(float(w) for w in workspace)
        self.r_min, self.r_max = self._reach_limits()
        self.q = np.clip(np.asarray(q0, float), self.q_lo, self.q_hi)
        if not self._allowed(self.fk()):
            raise ValueError(f"q0 = {tuple(q0)} puts the end effector at {self.fk().round(4).tolist()} m, "
                             f"outside the workspace {self.workspace} / reach [{self.r_min:.3f}, {self.r_max:.3f}] m")
        self.qdot = np.zeros(2)
        self._t = 0.0

    def _reach(self, q2: float) -> float:
        return float(np.sqrt(self.l1 ** 2 + self.l2 ** 2 + 2 * self.l1 * self.l2 * np.cos(q2)))

    def _reach_limits(self) -> tuple:
        """Radius range reachable within the elbow limits (r is monotone in |q2| on [0, π])."""
        lo, hi = self.q_lo[1], self.q_hi[1]
        candidates = [lo, hi] + ([0.0] if lo <= 0.0 <= hi else [])
        radii = [self._reach(c) for c in candidates]
        return min(radii), max(radii)

    def fk(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """End-effector position (m) for joint angles q (default: current)."""
        q1, q2 = self.q if q is None else q
        return np.array([self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2),
                         self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)])

    def jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        q1, q2 = self.q if q is None else q
        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
        return np.array([[-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
                         [self.l1 * c1 + self.l2 * c12, self.l2 * c12]])

    def _clamp_point(self, p: np.ndarray) -> np.ndarray:
        """Project p into the workspace box, then radially into the reachable annulus."""
        if self.workspace is not None:
            xmin, xmax, ymin, ymax = self.workspace
            p = np.array([np.clip(p[0], xmin, xmax), np.clip(p[1], ymin, ymax)])
        r = float(np.hypot(p[0], p[1]))
        if r > self.r_max:
            p = p * (self.r_max / r)
        elif 0.0 < r < self.r_min:
            p = p * (self.r_min / r)
        return p

    def _allowed(self, p: np.ndarray) -> bool:
        """p within WORKSPACE_TOL_M of the allowed region (box ∩ reachable annulus)."""
        return bool(np.linalg.norm(self._clamp_point(p) - p) <= self.WORKSPACE_TOL_M)

    def _apply(self, u: np.ndarray, dt: float) -> None:
        if dt == 0.0:
            self.qdot = np.zeros(2)
            return
        n_sub = max(1, int(np.ceil(dt / self.MAX_SUBSTEP_S - 1e-9)))
        h = dt / n_sub
        q_start = self.q.copy()
        for _ in range(n_sub):
            self._substep(u, h)
        self.qdot = (self.q - q_start) / dt      # the mean joint velocity actually realised
        self._t += dt

    def _substep(self, u: np.ndarray, h: float) -> None:
        p = self.fk()
        xdot = (self._clamp_point(p + u * h) - p) / h
        J = self.jacobian()
        qdot = J.T @ np.linalg.solve(J @ J.T + self.damping ** 2 * np.eye(2), xdot)
        peak = np.max(np.abs(qdot))
        if peak > self.max_joint_speed:
            qdot = qdot * (self.max_joint_speed / peak)
        q_new = np.clip(self.q + qdot * h, self.q_lo, self.q_hi)
        if self._allowed(self.fk(q_new)):
            self.q = q_new

    def _stop(self) -> None:
        self.qdot = np.zeros(2)

    def state(self) -> dict:
        return {**super().state(), "pos": self.fk().tolist(), "q": self.q.tolist(),
                "qdot": self.qdot.tolist(), "t": self._t}
