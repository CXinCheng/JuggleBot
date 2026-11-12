"""Robustness of the free-approach fingertip-ending solutions: landing offset x speed sweep (and approach-angle jitter)."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import search_end_tip2 as S

def job(args):
    fn, off, speed, daz, dpl = args
    d = json.loads(Path(fn).read_text()); cfg = dict(d["cfg"]); cfg["off"] = [off[0], off[1], 0.0]; cfg["speed"] = speed
    cfg["az"] += daz; cfg["el"] = min(cfg["el"] + dpl, 89.9)
    rig, a0, _ = S.run(np.array(d["x"]), P=S._P, cfg=cfg); m = S.metrics(rig, a0)
    ok = m["alpha_drift"] < 0.02 and m["beta_err"] < 0.04 and m["beta_speed"] < 0.05 and m["beta_contacts"] >= 2 and m["wrist"] == 0
    return args, ok, m["alpha_drift"], m["beta_err"]

def init():
    S._P = S.Problem("right", 10.0)

if __name__ == "__main__":
    out = {}
    for fn in sys.argv[1:]:
        grid = [(fn, (ox, oy), sp, 0, 0) for ox in (-.02, -.01, 0, .01, .02) for oy in (-.02, -.01, 0, .01, .02) for sp in (0.5, 1.5, 2.5)]
        grid += [(fn, (0, 0), 1.5, daz, dpl) for daz in (-20, -10, 0, 10, 20) for dpl in (-15, 0, 15)]
        with Pool(16, initializer=init) as pool:
            res = pool.map(job, grid)
        pos = [r for r in res[:75]]
        print(fn)
        for sp in (0.5, 1.5, 2.5):
            sub = [r for r in pos if r[0][2] == sp]
            print(f"  speed {sp}: offset-grid success {sum(r[1] for r in sub)}/25;  failures at", [(r[0][1]) for r in sub if not r[1]][:6])
        ang = res[75:]
        print("  approach-angle jitter (daz,dpl):", sum(r[1] for r in ang), "/", len(ang), "fail:", [(r[0][3], r[0][4]) for r in ang if not r[1]])
        out[fn] = [dict(args=[list(map(float, a)) if isinstance(a, tuple) else a for a in r[0][1:]], ok=bool(r[1]), alpha_drift=r[2], beta_err=r[3]) for r in res]
    Path("v7_three_ball_hold/results/end_tip2_robust.json").write_text(json.dumps(out))
