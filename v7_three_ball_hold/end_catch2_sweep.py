"""Sweep catch-position offsets and closure timing for the ending catch (end_catch2.py)."""
import sys, json, itertools
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from end_catch2 import run, score, Problem, frame_data
X = None; P = None; FD = None
def init(fn):
    global X, P, FD
    X = np.array(json.loads(Path(fn).read_text())["x"]); P = Problem("right", 10.0); FD = frame_data()
def job(args):
    ox, oy, tc, cd = args
    rig, a0, _ = run(P, X, t_close=tc, close_dur=cd, offset=(ox, oy), fd=FD)
    return args, score(rig, a0)
if __name__ == "__main__":
    fn = sys.argv[1]
    offs = [-0.02, -0.01, 0.0, 0.01, 0.02]
    timing = [(1.20, 0.10), (1.27, 0.10), (1.27, 0.20)]
    jobs = [(ox, oy, tc, cd) for ox in offs for oy in offs for tc, cd in timing]
    with Pool(18, initializer=init, initargs=(fn,)) as pool:
        res = pool.map(job, jobs)
    ok = sum(r[1]["ok"] for r in res)
    print("success", ok, "/", len(res))
    for tc, cd in timing:
        print(f"timing t_close={tc} dur={cd}")
        for ox in offs:
            print(" ox=%+.2f " % ox + " ".join("Y" if next(r for r in res if r[0] == (ox, oy, tc, cd))[1]["ok"] else "." for oy in offs))
    Path(fn.replace(".json", "_endcatch_sweep.json")).write_text(json.dumps([[list(a), s] for a, s in res]))
