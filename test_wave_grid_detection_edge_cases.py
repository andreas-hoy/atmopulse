import numpy as np
from backend_waves import detect_kysely_waves_grid
from test_wave_grid_detection import point_reference

cases = []

# 1) all-NaN season
cases.append(("all_nan", np.full(30, np.nan), 24.0, 19.0, True))
# 2) exactly-at-threshold values (boundary equality, warm >=)
t = np.array([24.0, 24.0, 24.0, 24.0, 19.0, 24.0, 24.0, 24.0, 30.0], dtype=float)
cases.append(("boundary_eq_warm", t, 24.0, 19.0, True))
# 3) exactly at p_drop boundary (should NOT break; drop_break is strict <)
t2 = np.array([25.0, 25.0, 25.0, 19.0, 25.0, 25.0], dtype=float)
cases.append(("boundary_drop_eq", t2, 24.0, 19.0, True))
# 4) long uninterrupted event
t3 = np.full(50, 30.0)
cases.append(("long_event", t3, 24.0, 19.0, True))
# 5) event ends exactly at season end (no trailing days to confirm break)
t4 = np.array([10.0, 10.0, 30.0, 30.0, 30.0, 30.0, 30.0], dtype=float)
cases.append(("ends_at_edge", t4, 24.0, 19.0, True))
# 6) two adjacent events separated by exactly enough of a gap
t5 = np.array([30, 30, 30, 5, 5, 30, 30, 30, 5], dtype=float)
cases.append(("two_events", t5, 24.0, 19.0, True))
# 7) cold direction sanity
t6 = np.array([-30, -30, -30, -10, -30, -30, -30, 5], dtype=float)
cases.append(("cold_basic", t6, -24.0, -19.0, False))
# 8) NaN gap splits an event mid-stream
t7 = np.array([30, 30, 30, np.nan, 30, 30, 30], dtype=float)
cases.append(("nan_gap_mid", t7, 24.0, 19.0, True))

all_ok = True
for name, temps, p_thresh, p_drop, is_warm in cases:
    n = len(temps)
    ref_int, ref_dur = point_reference(temps, p_thresh, p_drop, is_warm)
    grid_int, grid_dur = detect_kysely_waves_grid(
        temps.reshape(n, 1, 1), np.array([[p_thresh]]), np.array([[p_drop]]), is_warm,
    )
    grid_int, grid_dur = grid_int[:, 0, 0], grid_dur[:, 0, 0]
    ok = np.allclose(ref_int, grid_int, atol=1e-4) and np.array_equal(ref_dur, grid_dur)
    print(f"{name:18s}: {'OK' if ok else 'FAIL'}")
    if not ok:
        all_ok = False
        print("  temps:", temps)
        print("  ref_int:", ref_int, "ref_dur:", ref_dur)
        print("  grid_int:", grid_int, "grid_dur:", grid_dur)

assert all_ok, "edge-case mismatch"
print("ALL EDGE CASES PASS")
