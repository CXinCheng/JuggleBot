"""Robustness of the dynamic fingertip ending catch: landing offset x speed grid and approach-angle jitter."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import search_catch_tip as S
SPEEDS = (0.5, 1.5, 2.5, 3.5)
RELAX = True

def job(args):
    fn, off, speed, daz, dpl = args
    d = json.loads(Path(fn).read_text()); cfg = dict(d["cfg"]); cfg["off"] = [off[0], off[1], 0.0]; cfg["speed"] = speed
    cfg["az"] += daz; cfg["el"] = min(cfg["el"] + dpl, 89.9)
    rig, a0, bb, _ = S.run(np.array(d["x"]), P=S._P, cfg=cfg); m = S.metrics(rig, a0, bb)
    m["b_err"] = float(np.linalg.norm(np.array(m["b_pos"]) - S.B_TGT))
    relaxed = m["a_drift"] < 0.02 and m["b_speed"] < 0.05 and m["a_speed"] < 0.05 and m["nb"] >= 2 and "rf" in m["a_parts"] and m["wrist"] == 0 and np.linalg.norm(np.array(m["b_pos"]) - S.B_TGT) < 0.08 and min(m["b_pos"][2], 1) > -0.01
    return args, bool(relaxed if RELAX else S.ok(m)), m["a_drift"], m["b_err"]

def init():
    S._P = S.Problem("right", 10.0)

if __name__ == "__main__":
    out = {}
    for fn in sys.argv[1:]:
        grid = [(fn, (ox, oy), sp, 0, 0) for ox in (-.02, -.01, 0, .01, .02) for oy in (-.02, -.01, 0, .01, .02) for sp in SPEEDS]
        grid += [(fn, (0, 0), 2.5, daz, dpl) for daz in (-20, -10, 0, 10, 20) for dpl in (-15, 0, 15)]
        with Pool(16, initializer=init) as pool:
            res = pool.map(job, grid)
        n = 25 * len(SPEEDS); print(fn)
        for sp in SPEEDS:
            sub = [r for r in res[:n] if r[0][2] == sp]
            print(f"  speed {sp}: offset-grid success {sum(r[1] for r in sub)}/25; failures", [r[0][1] for r in sub if not r[1]][:8])
        ang = res[n:]; print("  angle jitter @2.5 m/s:", sum(r[1] for r in ang), "/", len(ang), "fail", [(r[0][3], r[0][4]) for r in ang if not r[1]])
        out[fn] = [dict(args=[list(a) if isinstance(a, tuple) else a for a in r[0][1:]], ok=r[1], a_drift=r[2], b_err=r[3]) for r in res]
    Path("v7_three_ball_hold/results/catch_tip_robust.json").write_text(json.dumps(out))
