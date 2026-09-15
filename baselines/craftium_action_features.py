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


# WorldMem's own convert_action_space() (minecraft_video_dataset.py:46-56) only ever writes 8 of
# ACTION_KEYS' 25 slots during pretraining data generation: hotbar.1(2), forward(11), back(12),
# left(13), right(14), cameraY(15), cameraX(16), drop(24). The other 17 slots' pretrained weight
# columns never received a real training signal (always-zero input throughout pretraining), so
# this reconstruction keeps them forced to 0 even where Craftium *could* supply a real value
# (sneak(18), aux1->sprint(19), dig->attack(21), slot_2-9->hotbar.2-9(3-10)) -- an explicit,
# controlled-variable choice (see docs/action_space_empirical_report.md and the Hypothesis A/B
# diagnostic work), not an oversight.
WORLDMEM_PRETRAINING_REACHABLE_SLOTS = np.zeros(25, dtype=bool)
WORLDMEM_PRETRAINING_REACHABLE_SLOTS[[2, 11, 12, 13, 14, 15, 16, 24]] = True

CRAFTIUM_ACTION_DIM_WORLDMEM_NATIVE = 25  # same width as CRAFTIUM_ACTION_DIM_ENRICHED, different
# column semantics: this one matches WorldMem's own ACTION_KEYS layout column-for-column, rather
# than appending 2 bespoke columns after Craftium's raw 23.


def augment_actions_to_worldmem_native_layout(
    action: np.ndarray, player_yaw: np.ndarray, player_pitch: np.ndarray
) -> np.ndarray:
    """Reconstructs Craftium's raw 23-dim action + player_yaw/player_pitch directly into
    WorldMem's OWN 25-slot VPT-style ACTION_KEYS layout (minecraft_video_dataset.py), instead of
    appending bespoke columns (contrast augment_actions_with_camera_features). Returns (T, 25)
    float32, column order == ACTION_KEYS. Only WORLDMEM_PRETRAINING_REACHABLE_SLOTS positions are
    ever written a nonzero value; the other 17 slots are forced to 0.0.

    cameraX(16)/cameraY(15) are SIGN-ONLY ({-1, 0, 1}), NOT the raw yaw/pitch magnitude --
    verified directly against convert_action_space (minecraft_video_dataset.py:48-51), WorldMem's
    own pretraining data never produces any value other than exactly -1, 0, or 1 for these two
    slots, so the pretrained weight columns were calibrated for inputs of that exact magnitude.
    Feeding real degree magnitudes (which can be 6 deg or much larger) would be several times
    outside that calibrated range -- actively counterproductive for informed-init, which is the
    whole point of using this native layout instead of the bespoke append. Every other reachable
    slot (forward/back/left/right/hotbar.1) is verified to be strictly {0,1} on BOTH sides
    (convert_action_space only ever writes literal 1 to them; Craftium's source columns are
    provably pure bool from MultiDiscreteActionWrapper.multihot()) -- a safe direct pass-through,
    no scaling needed.

    hotbar.1(2) <- Craftium slot_1(10) caveat: Craftium's slot_1 can only ever be set together
    with `place`(8) -- the MultiDiscreteActionWrapper action-group list only offers combined
    "place+slot_N" choices, never a standalone slot-select (utils/action_util.py) -- unlike
    WorldMem's native hotbar.1, a pure slot-switch with no implied placing action. Kept as the
    closest available mapping; almost certainly ~always 0 in practice since dig/place is rarely
    exercised in this data.
    """
    n = action.shape[0]
    vec = np.zeros((n, 25), dtype=np.float32)

    vec[:, 11] = action[:, 0]   # forward   <- Craftium forward
    vec[:, 12] = action[:, 1]   # back      <- Craftium backward
    vec[:, 13] = action[:, 2]   # left      <- Craftium left
    vec[:, 14] = action[:, 3]   # right     <- Craftium right
    vec[:, 2] = action[:, 10]   # hotbar.1  <- Craftium slot_1 (place-coupled caveat, see docstring)

    cam = compute_clean_camera_deltas(player_yaw, player_pitch, action)  # (T,2) = [yaw_delta, pitch_delta]
    vec[:, 16] = np.sign(cam[:, 0])  # cameraX <- sign(yaw_delta)   -- {-1,0,1} only, NOT magnitude
    vec[:, 15] = np.sign(cam[:, 1])  # cameraY <- sign(pitch_delta) -- {-1,0,1} only, NOT magnitude

    # drop(24) intentionally left 0.0 -- doubly unreachable: Craftium's own drop (base_actions
    # "drop") is never reachable via MultiDiscreteActionWrapper (no action group ever contains
    # "drop" alone), and separately WorldMem's own drop-reachability needs a MineDojo raw signal
    # Craftium has no equivalent of -- nothing to map, no conflict.

    vec[:, ~WORLDMEM_PRETRAINING_REACHABLE_SLOTS] = 0.0  # defensive re-assertion, not just docs
    return vec


# Identity map: source (Craftium-native-layout) and destination (WorldMem's own pretrained
# checkpoint) columns are now the SAME index, since augment_actions_to_worldmem_native_layout
# already places every value in WorldMem's own ACTION_KEYS column position -- no reindexing like
# the 23-dim path's INFORMED_INIT_ACTION_MAP needs. Deliberately excludes jump(17)/sneak(18)/
# sprint(19)/attack(21)/hotbar.2-9(3-10) -- these are NOT informed-init candidates: WorldMem's own
# pretraining never wrote to them either (see WORLDMEM_PRETRAINING_REACHABLE_SLOTS), so their
# pretrained weight columns are uninformative, same category as drop.
WORLDMEM_NATIVE_INFORMED_INIT_ACTION_MAP = {i: i for i in [2, 11, 12, 13, 14, 15, 16, 24]}


if __name__ == "__main__":
    # Standalone structural verification, no training needed -- run on real Craftium episodes.
    # See the "Reconstruct Craftium actions into WorldMem's own native 25-slot layout" plan's
    # Verification section. Verified once (2026-09) on 20 real episodes: all assertions pass;
    # cameraX/cameraY only ever take values in {-1,0,1}, drop(24) is always exactly 0, and the 5
    # direct pass-through columns (hotbar.1, forward, back, left, right) match Craftium's own
    # source columns exactly.
    import glob
    import os

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from finetune_worldmem_data import DATASET_ROOT

    levels = sorted(glob.glob(os.path.join(DATASET_ROOT, "level_*")))[:20]
    print(f"Checking {len(levels)} real episodes...")

    for lvl in levels:
        npz = np.load(os.path.join(lvl, "data.npz"))
        action = npz["action"]
        out = augment_actions_to_worldmem_native_layout(action, npz["player_yaw"], npz["player_pitch"])

        assert out.shape == (action.shape[0], 25), f"{lvl}: bad shape {out.shape}"
        assert np.all(out[:, ~WORLDMEM_PRETRAINING_REACHABLE_SLOTS] == 0.0), f"{lvl}: unreachable slot nonzero"
        assert np.array_equal(out[:, [2, 11, 12, 13, 14]], action[:, [10, 0, 1, 2, 3]]), (
            f"{lvl}: pass-through columns mismatch"
        )
        assert set(np.unique(out[:, 16]).tolist()) <= {-1.0, 0.0, 1.0}, f"{lvl}: cameraX out of range"
        assert set(np.unique(out[:, 15]).tolist()) <= {-1.0, 0.0, 1.0}, f"{lvl}: cameraY out of range"
        assert set(np.unique(out[:, 24]).tolist()) == {0.0}, f"{lvl}: drop nonzero"

    print("All structural assertions passed on all 20 episodes.")

    lvl = levels[0]
    npz = np.load(os.path.join(lvl, "data.npz"))
    out = augment_actions_to_worldmem_native_layout(npz["action"], npz["player_yaw"], npz["player_pitch"])
    print(f"\n{lvl} sanity check:")
    print(f"  drop(24).sum() = {out[:,24].sum()} (expect exactly 0)")
    print(f"  hotbar.1(2).sum() = {out[:,2].sum()} (expect very small/near 0)")
    print(f"  cameraX(16) unique values: {np.unique(out[:,16])}")
    print(f"  cameraY(15) unique values: {np.unique(out[:,15])}")
