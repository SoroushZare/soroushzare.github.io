"""RunLogger crash safety: a process killed mid-run (liblsl segfault, OOM kill) leaves a run that
loads, with at most the chunk being written lost and the sample and timestamp files consistent."""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from neural_cursor.config import DEFAULT
from neural_cursor.data_engine import SessionStore
from neural_cursor.logger import RunLogger

N_CHUNKS, CHUNK, N_CURSOR = 2000, 5, 200


def test_killed_run_is_complete_and_loadable(tmp_path):
    script = textwrap.dedent(f"""
        import os, numpy as np
        from neural_cursor.config import DEFAULT
        from neural_cursor.logger import RunLogger
        lg = RunLogger({str(tmp_path)!r}, DEFAULT, tag="crash")
        for i in range({N_CHUNKS}):
            lg.eeg(np.full(({CHUNK}, 32), i, np.float32), i * {CHUNK} + np.arange({CHUNK}, dtype=float))
        for i in range({N_CURSOR}):
            lg.cursor(float(i), (0.0, 0.0), (0.0, 0.0), 0.5, None, 0.0)
        os._exit(1)                                       # no close(), no flush: a hard crash
    """)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-c", script], cwd=root, check=False, timeout=60)
    (run,) = [os.path.join(tmp_path, d) for d in os.listdir(tmp_path)]
    x, ts = RunLogger.load_eeg(run, 32)
    assert len(x) == len(ts) == N_CHUNKS * CHUNK                    # nothing was held in a buffer
    rows = SessionStore.read_cursor(os.path.join(run, "cursor.csv"))
    assert len(rows) == N_CURSOR
    assert len(SessionStore(str(tmp_path)).load(run).eeg) == N_CHUNKS * CHUNK


def test_cut_off_files_are_truncated_to_their_common_length(tmp_path):
    lg = RunLogger(str(tmp_path), DEFAULT, tag="cut")
    lg.eeg(np.ones((10, 32), np.float32), np.arange(10.0))
    lg.close()
    with open(os.path.join(lg.dir, "eeg.f32"), "ab") as f:          # a chunk half-written at the crash
        f.write(np.ones(32 * 3 + 7, np.float32).tobytes())
    with pytest.warns(RuntimeWarning, match="cut off"):
        x, ts = RunLogger.load_eeg(lg.dir, 32)
    assert x.shape == (10, 32) and len(ts) == 10
