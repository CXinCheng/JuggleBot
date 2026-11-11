"""Run the 2+1-start cascade (see cascade21.py). Example: uv run python v7_three_ball_hold/run_cascade21.py --throws 8"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cascade21 import TwoPlusOneCascade, HOLD

ap = argparse.ArgumentParser()
ap.add_argument("--throws", type=int, default=6); ap.add_argument("--timeout", type=float, default=12.0)
ap.add_argument("--t-open-age", type=float, default=None); ap.add_argument("--hold", default=str(HOLD))
ap.add_argument("--verbose", action="store_true"); ap.add_argument("--guard", action="store_true")
a = ap.parse_args()
c = TwoPlusOneCascade(hold_file=a.hold, t_open_age=a.t_open_age, verbose=a.verbose)
t0 = time.time()
res = c.run_continuous(throws=a.throws, timeout=a.timeout, guard=a.guard)
print("wall %.1fs" % (time.time() - t0))
for k in ("success", "error", "throws", "catches", "validated_flights", "time", "ball_contacts", "drops", "min_height", "min_surface_gap_m"):
    print(k, res.get(k))
print("events:")
for e in c.events[:40]:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k in ("time", "kind", "side", "ball", "count", "error", "timing_error", "score", "reason")})
