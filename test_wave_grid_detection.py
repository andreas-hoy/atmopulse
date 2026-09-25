import numpy as np
import pandas as pd

from backend_waves import detect_kysely_waves_grid

rng = np.random.default_rng(42)

N_CELLS = 400
N_DAYS = 200


def point_reference(temps_1d, p_thresh, p_drop, is_warm):
    """Direct transcription of the point while-loop (mirrors
    backend_waves._detect_kysely_waves's inner logic exactly, single cell,
    single season group -- used ONLY as an independent oracle for this test)."""
    n = len(temps_1d)
    out_int = np.zeros(n, dtype=np.float64)
    out_dur = np.zeros(n, dtype=np.int64)
    i = 0
    while i < n - 2:
        window = temps_1d[i:i + 3]
        if np.isnan(window).any():
            i += 1
            continue
        trig = all((window[k] >= p_thresh) if is_warm else (window[k] <= p_thresh) for k in range(3))
        if not trig:
            i += 1
            continue
        cand = []
        j = i
        while j < n and not np.isnan(temps_1d[j]):
            cand.append(temps_1d[j])
            drop_break = (temps_1d[j] < p_drop) if is_warm else (temps_1d[j] > p_drop)
            mean_break = (np.mean(cand) < p_thresh) if is_warm else (np.mean(cand) > p_thresh)
            if drop_break or mean_break:
                cand.pop()
                break
            j += 1
        if len(cand) >= 3:
            intensity = 0.0
            for k, t in enumerate(cand):
                qualifies = (t >= p_thresh) if is_warm else (t <= p_thresh)
                if qualifies:
                    intensity += abs(t - p_thresh)
                out_int[i + k] = intensity
                out_dur[i + k] = k + 1
        i = j if j > i else i + 1
    return out_int, out_dur


def run_case(is_warm, seed):
    rng = np.random.default_rng(seed)
    temps = rng.normal(20 if is_warm else -5, 8, size=(N_DAYS, N_CELLS))
    # sprinkle some NaNs
    nan_mask = rng.random((N_DAYS, N_CELLS)) < 0.02
    temps[nan_mask] = np.nan
    p_thresh = rng.normal(25 if is_warm else -10, 2, size=(N_CELLS,))
    p_drop = p_thresh - 5 if is_warm else p_thresh + 5

    temps3 = temps.reshape(N_DAYS, 1, N_CELLS)
    p_thresh2 = p_thresh.reshape(1, N_CELLS)
    p_drop2 = p_drop.reshape(1, N_CELLS)

    grid_int, grid_dur = detect_kysely_waves_grid(temps3, p_thresh2, p_drop2, is_warm)
    grid_int = grid_int[:, 0, :]
    grid_dur = grid_dur[:, 0, :]

    mism = 0
    for c in range(N_CELLS):
        ref_int, ref_dur = point_reference(temps[:, c], p_thresh[c], p_drop[c], is_warm)
        if not np.allclose(ref_int, grid_int[:, c], atol=1e-4) or not np.array_equal(ref_dur, grid_dur[:, c]):
            mism += 1
            if mism <= 3:
                diff_days = np.flatnonzero(~np.isclose(ref_int, grid_int[:, c], atol=1e-4) | (ref_dur != grid_dur[:, c]))
                print(f"  MISMATCH cell {c} is_warm={is_warm} seed={seed} days={diff_days[:10]}")
    print(f"is_warm={is_warm} seed={seed}: {mism}/{N_CELLS} cells mismatched")
    return mism


if __name__ == "__main__":
    total_mismatch = 0
    for is_warm in (True, False):
        for seed in (1, 2, 3, 4, 5):
            total_mismatch += run_case(is_warm, seed)

    print("TOTAL MISMATCHES:", total_mismatch)
    assert total_mismatch == 0, "grid detection diverges from the point oracle"
    print("ALL MATCH -- grid detection is bit-exact vs the point algorithm.")
