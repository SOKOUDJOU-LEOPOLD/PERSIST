"""Enriched Craftium action features: adds a clean, near-exact camera-turn signal derived from
`player_yaw`/`player_pitch`, alongside Craftium's raw 23-dim boolean action vector.

WHY THIS EXISTS (see docs/action_space_empirical_report.md and the session's engine-source
investigation for the full derivation): Craftium's raw camera action is a boolean
(mouse_x-/x+/y-/y+, 0 or 1) that maps to a REAL yaw/pitch change verified to vary frame-to-frame
(ramping like 5.15 deg, 2.73 deg, 2.53 deg... rather than a fixed amount) when measured from the
`cam_dir` field -- because `cam_dir` reflects the RENDERED camera, which is a smoothed/lagged
version of the true, instantaneous player orientation. `player_yaw`/`player_pitch`, by contrast,
change by an exactly fixed amount (measured: 6.40 deg) every time the corresponding action is
truly active -- much closer to WorldMem's own pretraining camera semantics (a small number of
discrete, FIXED-magnitude values), and a much richer signal than the boolean input itself.

CORRECTING AN EARLIER PROJECT FINDING, not contradicting it: finetune_worldmem_data.py's own
docstring documents player_yaw as "stale ~62% of the time" and player_pitch as "frozen 96% of the
time" -- both numbers are correct as raw statistics, but conflate two different situations.
Verified directly on real data (level_001): yaw is frozen only 6.2% of the time when a camera
action (mouse_x) is genuinely active (i.e. 93.8% reliable exactly when it matters), and 98.9%
frozen when camera action is inactive -- which is CORRECT behavior (nothing should change if
nothing was commanded), not a bug. Pitch shows the same pattern once conditioned on mouse_y
specifically (14.3% frozen when mouse_y is active, vs. 99.5% when it's not) -- the earlier 96%
figure was diluted by lumping in frames where only mouse_x fired and pitch correctly held still.
So: these fields are NOT globally unreliable, they are reliable exactly when gated on their own
axis's own action -- which is exactly what this module does, so no stale-frame heuristic is
needed at all (no interpolation, no forward-fill -- just gate each axis's delta on its own
action).

`cam_dir`-derived pitch/yaw remains the correct choice for POSE conditioning (what the video
frames actually show) -- this module produces an ACTION feature, a different, additional signal,
not a replacement for existing pose handling.
"""
import numpy as np


def wrap_angle_deg(delta: np.ndarray) -> np.ndarray:
    """Wrap an angle difference to [-180, 180)."""
    return (delta + 180) % 360 - 180


def compute_clean_camera_deltas(player_yaw: np.ndarray, player_pitch: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Returns (T, 2) float32 array: [yaw_delta, pitch_delta] for frames 1..T-1 (frame 0 gets 0,0
    since there's no prior frame to diff against). Each axis's delta is GATED on that axis's own
    action being active that frame -- inactive frames get exactly 0.0, which is both correct
    (nothing was commanded) and avoids trusting a frozen/stale raw value as if it were a real
    zero-movement measurement.

    Craftium action indices (utils/action_util.py): 19=mouse_x-, 20=mouse_x+, 21=mouse_y-, 22=mouse_y+.
    """
    n = len(player_yaw)
    yaw_delta = np.zeros(n, dtype=np.float32)
    pitch_delta = np.zeros(n, dtype=np.float32)

    raw_yaw_delta = wrap_angle_deg(np.diff(player_yaw.astype(np.float64)))
    raw_pitch_delta = wrap_angle_deg(np.diff(player_pitch.astype(np.float64)))

    mouse_x_active = (action[1:, 19] != 0) | (action[1:, 20] != 0)
    mouse_y_active = (action[1:, 21] != 0) | (action[1:, 22] != 0)

    yaw_delta[1:] = np.where(mouse_x_active, raw_yaw_delta, 0.0)
    pitch_delta[1:] = np.where(mouse_y_active, raw_pitch_delta, 0.0)

    return np.stack([yaw_delta, pitch_delta], axis=1).astype(np.float32)  # (T, 2)


def augment_actions_with_camera_features(action: np.ndarray, player_yaw: np.ndarray, player_pitch: np.ndarray) -> np.ndarray:
    """Concatenates the 2 clean camera-delta features onto Craftium's raw 23-dim action vector,
    producing a (T, 25) array -- the raw boolean flags are kept (not replaced), so no information
    is lost, only added. Column order: [0..22] = original Craftium action, [23] = yaw_delta,
    [24] = pitch_delta."""
    cam_features = compute_clean_camera_deltas(player_yaw, player_pitch, action)
    return np.concatenate([action.astype(np.float32), cam_features], axis=1)  # (T, 25)


CRAFTIUM_ACTION_DIM_ENRICHED = 25  # 23 original + yaw_delta + pitch_delta
