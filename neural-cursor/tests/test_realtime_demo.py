"""Real-time demo tests: the champion chain end to end over a real LSL outlet on localhost.

One short run (15 s live after the demo's default 24 min of simulated L1 calibration) is shared by the end-to-end checks: it
must keep p95 latency under 50 ms at the control rate, drive both devices from one decoded command, and
move them in the imagined body direction on y for every scored segment.  The decoder's inputs are
recorded through the demo's own tap to assert the pure-EEG rule.  The remaining tests are fast unit
checks of the geometry, the fan-out effector and the segment scoring.
"""
import numpy as np
import pytest

from neural_cursor import realtime_demo as rd
from neural_cursor.config import DEFAULT
from neural_cursor.effectors import NullEffector, PlanarArm2Link, ScreenCursor

RUN_S = 15.0           # timed live run: FEET, HANDS, then RIGHT once x is earned (else FEET again)
SEGMENT_S = 5.0        # the demo's default: 1.5-s lockout + 3.5 s scored per segment


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    """The shared end-to-end run, with every decoder input's shape and finiteness recorded."""
    seen = {"shapes": set(), "finite": True, "good": set()}
    original = rd.DecoderTap.predict

    def spy(self, window, good=None):
        seen["shapes"].add(np.shape(window))
        seen["finite"] &= bool(np.isfinite(window).all())
        seen["good"].add(None if good is None else np.shape(good))
        return original(self, window, good=good)

    calibrate = rd.calibrate

    def keep_decoder(demo, cfg, subject):
        decoder, summary = calibrate(demo, cfg, subject)
        seen["decoder"], seen["subject"], seen["cfg"] = decoder, subject, cfg
        return decoder, summary

    mp = pytest.MonkeyPatch()
    mp.setattr(rd.DecoderTap, "predict", spy)
    mp.setattr(rd, "calibrate", keep_decoder)
    try:
        # The demo's default person and calibration: shorter calibrations give a y model whose 3.5-s
        # segment means are right only ≈ 85 % of the time (offline check), which would make this test
        # about sampling luck rather than about the chain's sign conventions.
        demo = rd.DemoConfig(seconds=RUN_S, segment_s=SEGMENT_S, effector="both")
        res = rd.run_demo(demo)
    finally:
        mp.undo()
    path = rd.write_report(res, str(tmp_path_factory.mktemp("demo") / "realtime_demo.md"))
    return res, seen, path


def test_latency_rate_and_stream_health(demo_run):
    res, _, _ = demo_run
    st = res["live"]["stats"]
    rate = res["control_hz"]                                  # the deployed loop's 16 Hz (loop_config)
    assert rate == 16.0
    expected = int(RUN_S * rate)
    assert st["n_expected"] == expected
    assert st["n_updates"] >= 0.9 * expected, st
    assert st["latency_ms"]["p95"] < 50.0, st
    assert st["latency_ms"]["p50"] > 0.0, st             # causal timestamps: never in the future
    assert st["aborted"] is None and st["dropped_samples"] == 0 and st["n_invalid"] == 0, st
    assert 0.9 * rate <= res["live"]["update_rate_hz"] <= 1.1 * rate


def test_devices_move_in_the_imagined_y_direction(demo_run):
    res, _, _ = demo_run
    segs = [s for s in res["live"]["segments"] if s["scored"] and s["axis"] == "y"]
    assert {s["class"] for s in segs} == {"FEET", "HANDS"}, res["live"]["segments"]
    for s in segs:
        assert s["decoded_agrees"], s
        assert s["agrees"] == {"cursor": True, "arm": True}, s
    k, n = res["live"]["summary"]["y"]["arm"]
    assert k == n == len(segs)


def test_calibration_uses_fixation_probes_only_and_decoder_sees_only_eeg(demo_run):
    res, seen, _ = demo_run
    cal = res["calibration"]
    assert cal["sources"] == ["L1"]
    assert cal["n_by_class"]["FEET"] > 0 and cal["n_by_class"]["HANDS"] > 0
    n_win = int(round(res["window_s"] * DEFAULT.hardware.fs))
    assert seen["shapes"] == {(n_win, DEFAULT.hardware.n_channels)} and seen["finite"]
    assert seen["good"] == {(DEFAULT.hardware.n_channels,)}  # the loop's own validity mask, nothing else


def test_report_is_written(demo_run):
    _, _, path = demo_run
    text = open(path).read()
    for heading in ("## Calibration", "## Live loop", "## Does the end effector move in the imagined body direction?"):
        assert heading in text
    assert "latency p50 / p95 / max" in text and "FEET (y)" in text and "HANDS (y)" in text


def test_arm_start_pose_inverse_kinematics():
    q = rd.arm_joint_angles(*rd.ARM_LINKS_M, (0.0, 0.30))
    arm = PlanarArm2Link(*rd.ARM_LINKS_M, q0=q)
    np.testing.assert_allclose(arm.fk(), [0.0, 0.30], atol=1e-12)
    assert q[1] >= 0.0                                      # elbow-up branch, inside the default limits
    with pytest.raises(ValueError):
        rd.arm_joint_angles(*rd.ARM_LINKS_M, (0.0, 0.60))   # beyond l1 + l2


def test_tee_effector_forwards_one_command_and_closes_children():
    cursor = ScreenCursor(101, 101, gain_px_s=10.0, backend="log")
    null = NullEffector()
    tee = rd.TeeEffector({"cursor": cursor, "null": null})
    tee.send(np.array([2.0, 0.5]), 1.0)                    # clipped once, each child applies its own speed
    np.testing.assert_allclose(cursor.pos, [50.0 + 10.0, 50.0 - 5.0])
    np.testing.assert_allclose(null.state()["v"], [1.0, 0.5])
    np.testing.assert_allclose(tee.task_position(), cursor.task_position())
    tee.close()
    with pytest.raises(RuntimeError):
        cursor.send(np.zeros(2), 0.1)


def test_score_segments_counts_displacement_after_the_lockout():
    """Synthetic ticks: the device moves up in segment 0 only after the lockout, and down in segment 1;
    motion before the lockout must not count, and a too-short tail is not scored."""
    seg_s, dt = 4.0, 0.1
    t = np.arange(0.0, 8.5, dt)
    ys = np.cumsum(np.where(t < seg_s, np.where(t >= rd.SCORE_LOCKOUT_S, 1.0, -5.0), -1.0)) * dt
    ticks = [(ti, int(ti // seg_s), {"arm": np.array([0.0, y])}) for ti, y in zip(t, ys)]
    ticks.append((8.5, -1, {"arm": np.array([0.0, ys[-1] - 0.1])}))
    decoded = [(ti, 0.0, 0.5 if ti < seg_s else -0.5) for ti in t]
    out = rd.score_segments(ticks, decoded, ("FEET", "HANDS"), seg_s, ["arm"])
    assert [s["class"] for s in out] == ["FEET", "HANDS", "FEET"]
    assert out[0]["agrees"]["arm"] and out[0]["decoded_agrees"] and out[0]["displacement"]["arm"][1] > 0
    assert out[1]["agrees"]["arm"] and out[1]["decoded_agrees"]
    assert not out[2]["scored"] and not out[2]["agrees"]["arm"]
    assert rd.summarise(out, ["arm"]) == {"y": {"arm": [2, 2], "decoded": [2, 2]}, "cross_axis_drift": {"y": 0.0}}
    assert out[0]["cross_axis_mean"] == 0.0                     # x while imagining feet


def test_demo_config_validation():
    with pytest.raises(ValueError):
        rd.DemoConfig(effector="robot")
    with pytest.raises(ValueError):
        rd.DemoConfig(segment_s=2.0)                        # no room for the lockout plus a scored span
    with pytest.raises(ValueError, match="dry_penalty"):
        rd.DemoConfig(dry_penalty=1.0)
    with pytest.raises(ValueError, match="calibration_minutes"):
        rd.DemoConfig(calibration_minutes=0.3)


@pytest.mark.parametrize("args", [["--seconds", "0"], ["--segment-s", "2"], ["--calibration-minutes", "0.3"],
                                  ["--dry-penalty", "1.0"]])
def test_demo_cli_turns_bad_arguments_into_usage_errors(args, capsys):
    """Review: every invalid argument was a raw traceback (some only after the whole calibration)."""
    with pytest.raises(SystemExit) as exc:
        rd.main(args)
    assert exc.value.code == 2 and "error" in capsys.readouterr().err


def test_one_hand_imagery_does_not_drive_y_after_x_is_earned(demo_run):
    """Review: one-hand imagery read as HANDS on y (a LEFT segment moved the arm 9.4 cm down), so an arm
    using sustained y for grasp and axis switches would act on lateral imagery.  The earned x refits y
    with one-hand windows as neutral: on fresh probes of the same person the mean y output on one-hand
    windows is far from the HANDS one, and the live run reports the cross-axis drift."""
    res, seen, _ = demo_run
    assert res["calibration"]["x_enabled"] and res["calibration"]["neutral_one_hand"]
    import tempfile
    from neural_cursor.data_engine import body_labels
    from neural_cursor.protocol import Block
    from neural_cursor.simulate_champion import ChampionSimConfig, record_fixation
    sim = ChampionSimConfig(days=1, users=1, pop=0, label_hop_s=0.5)
    with tempfile.TemporaryDirectory() as root:
        ls, refs, _ = record_fixation(seen["cfg"], sim, seen["subject"], 1, Block("fixation_probe", 240, seed=99), 5,
                                      "heldout", root, 2e9)
    dec = seen["decoder"].ungated()
    dec.set_reference(reference=next(iter(refs.values())))
    y_out, y_lab = dec.predict_many(ls.windows)[:, 1], body_labels(ls)[:, 1]
    one_hand, hands = float(np.nanmean(y_out[y_lab == 0])), float(np.nanmean(y_out[y_lab < 0]))
    assert hands < -0.1 and abs(one_hand) < 0.5 * abs(hands), (one_hand, hands)
    drift = res["live"]["summary"]["cross_axis_drift"]
    assert set(drift) >= {"y"} and all(np.isfinite(v) for v in drift.values())
