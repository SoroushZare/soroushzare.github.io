"""Body coordinates and the screen mappings that turn body velocity into cursor velocity.

The champion decoder reads *body-part intent*, not a screen direction: its output is a velocity in body
coordinates, y = feet (+) vs both hands (−), x = right hand (+) vs left hand (−).  The cursor moves by
``v_screen = g · M · v_body`` with an orthogonal 2×2 mapping ``M``.  Normal use and all training run
under the identity.  The held-out mappings (rot+90, rot−90, mirror_x, mirror_y) are used only in test
blocks: they are what lets ``certify.py`` tell a decoder that reads the body from one that reads the
screen.  Under the identity the two readings make identical predictions — the person who wants the
cursor to go up imagines feet, and the screen shows a target above the cursor — so no amount of
identity data can separate them.  Under a rotation by ±90° the person must imagine feet to move the
cursor sideways, so body intent and the on-screen direction become orthogonal on every trial.

Why the regressors are built here: the body intent a person must produce to reach a target is
``M⁻¹ (target − cursor)``, while a decoder that learned the screen under the identity would output
``M_train⁻¹ (target − cursor) = target − cursor``.  Both are unit vectors in the decoder's output frame,
so a regression of the output on both measures how much of it follows each (``certify.certify_mapping``).

Pure-EEG rule: this module is geometry only; it never sees a signal.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

_MATRICES: Dict[str, np.ndarray] = {
    "identity": np.array([[1.0, 0.0], [0.0, 1.0]]),
    "rot+90": np.array([[0.0, -1.0], [1.0, 0.0]]),    # counter-clockwise: feet (body +y) → cursor left
    "rot-90": np.array([[0.0, 1.0], [-1.0, 0.0]]),    # clockwise: feet (body +y) → cursor right
    "mirror_x": np.array([[-1.0, 0.0], [0.0, 1.0]]),  # right hand moves the cursor left
    "mirror_y": np.array([[1.0, 0.0], [0.0, -1.0]]),  # feet move the cursor down
}
MAPPING_NAMES: Tuple[str, ...] = tuple(_MATRICES)
TRAINING_MAPPING = "identity"
"""The only mapping any decoder (deployed or shadow) is ever trained under."""
HELDOUT_MAPPINGS: Tuple[str, ...] = ("rot+90", "rot-90", "mirror_x", "mirror_y")
"""Test-only mappings.  rot±90 is primary: it decorrelates body and screen on both axes within one
block, but under one rotation the screen direction on an axis equals ± the other axis's body intent,
so cross-axis body coupling is only separable from screen reading with BOTH rotations in a test unit
(``certify``; the protocol runs rot+90 and rot−90 every session).  A mirror flips one axis and leaves
the other identical (body = ±screen on each), so it identifies nothing per axis; mirrors stay valid
mappings but are not scheduled."""

BODY_VECTORS: Dict[str, Tuple[float, float]] = {
    "FEET": (0.0, 1.0),
    "HANDS": (0.0, -1.0),
    "RIGHT": (1.0, 0.0),
    "LEFT": (-1.0, 0.0),
}
"""Unit body velocity of each imagery class (diagonals are two-part cues: sums of two of these)."""

DIRECTION_EPS = 1e-9
"""Below this distance (task units) the cursor is on the target and the direction to it is undefined."""


class Mapping:
    """One named screen mapping ``M`` (orthogonal, so ``M⁻¹ = Mᵀ``).

    Vectors are rows: a ``(2,)`` vector or an ``(n, 2)`` array of them; the return has the same shape.
    """

    def __init__(self, name: str):
        if name not in _MATRICES:
            raise ValueError(f"unknown mapping {name!r}; expected one of {MAPPING_NAMES}")
        self.name = name
        self.M = _MATRICES[name].copy()
        self.M.setflags(write=False)

    @property
    def heldout(self) -> bool:
        """True for the test-only mappings, which no decoder may ever be trained under."""
        return self.name in HELDOUT_MAPPINGS

    @staticmethod
    def _rows(v) -> np.ndarray:
        v = np.asarray(v, float)
        if v.shape[-1] != 2 or v.ndim > 2:
            raise ValueError(f"expected a (2,) vector or an (n, 2) array, got shape {v.shape}")
        return v

    def body_to_screen(self, v_body) -> np.ndarray:
        """Screen velocity ``M v`` for body velocity ``v`` (row form: ``v Mᵀ``)."""
        return self._rows(v_body) @ self.M.T

    def screen_to_body(self, v_screen) -> np.ndarray:
        """Body velocity ``M⁻¹ u = Mᵀ u`` that would produce screen velocity ``u`` (row form: ``u M``)."""
        return self._rows(v_screen) @ self.M

    def __repr__(self) -> str:
        return f"Mapping({self.name!r})"


def unit_directions(targets, cursors) -> np.ndarray:
    """Unit screen vectors ``(target − cursor)/‖·‖`` per row; zero where the cursor is on the target
    (no direction exists there, and a zero regressor row then informs only the intercept)."""
    d = np.asarray(targets, float) - np.asarray(cursors, float)
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError(f"targets and cursors must be (n, 2), got {d.shape}")
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return np.where(norm > DIRECTION_EPS, d / np.maximum(norm, DIRECTION_EPS), 0.0)


def certification_regressors(targets, cursors, mapping: Mapping,
                             trained_on: str = TRAINING_MAPPING) -> Tuple[np.ndarray, np.ndarray]:
    """``(body_intent, screen_dir_in_body)`` for windows of a mapping block, both in the decoder's
    output frame.

    body_intent        ``M⁻¹ u`` with ``u`` the unit screen direction to the target: the body velocity
                       that moves the cursor toward the target under the block's mapping.
    screen_dir_in_body ``M_train⁻¹ u``: what a decoder that learned the screen under its training
                       mapping would output on the same window.
    ``targets`` / ``cursors`` are the displayed positions at each window's end (the cursor as the
    person sees it).
    """
    u = unit_directions(targets, cursors)
    return mapping.screen_to_body(u), Mapping(trained_on).screen_to_body(u)


DIAGONAL_DIRECTIONS: np.ndarray = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]) / np.sqrt(2.0)
"""Unit screen directions of the four diagonal targets, one per quadrant (signs (+,+), (−,+), (−,−),
(+,−)).  Held-out blocks use them because under rot±90 a 1-D y person moves the cursor only
horizontally: the reachable horizontal offset is the body regressor, and the unreachable vertical
offset — the screen regressor of the y output — must vary independently of it.  With targets on the
horizontal axis that screen regressor would be 0 on every trial and β_screen unidentified; with
random ring positions the two offsets are uncorrelated only on average.  Independent, balanced signs of
(dx, dy) make them uncorrelated by design, on both axes and under either rotation."""
DIAGONAL_DIRECTIONS.setflags(write=False)


def diagonal_targets(distance: float) -> np.ndarray:
    """(4, 2) targets at ``distance`` from the centre along the four diagonals (``DIAGONAL_DIRECTIONS``
    order), all drawn on every trial so the display is symmetric and the goal posterior has four
    candidates."""
    if not distance > 0:
        raise ValueError(f"distance must be > 0, got {distance}")
    return DIAGONAL_DIRECTIONS * float(distance)


def quadrant_schedule(n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Goal quadrant (index into ``DIAGONAL_DIRECTIONS``) per trial: every consecutive run of four
    trials is a random permutation of the four quadrants.  Balanced within every 4 trials, so the signs
    of dx and dy are independent and equally frequent in any block, whatever its length (a cut-off final
    run leaves at most 3 trials of imbalance), and the order is unpredictable within a run."""
    if n_trials < 0:
        raise ValueError(f"n_trials must be >= 0, got {n_trials}")
    runs = [rng.permutation(len(DIAGONAL_DIRECTIONS)) for _ in range(-(-n_trials // len(DIAGONAL_DIRECTIONS)))]
    return np.concatenate(runs)[:n_trials] if runs else np.zeros(0, int)
