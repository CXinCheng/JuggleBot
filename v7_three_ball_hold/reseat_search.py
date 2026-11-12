import sys, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from reseat import *

X = None; P = None; GL = None
def init(fn):
    global X, P, GL
    X = np.array(json.loads(Path(fn).read_text())["x"]); P = Problem("right", 10.0)
    t, ps, rot, ctrl, acc = load_rec(); GL = g_local_series(t, rot, acc)
def ev(p):
    rig, _ = run_reseat(P, X, p, t_end=1.50, gl=GL)
    err, sp, n, fs, parts = score(rig)
    return err, sp, n, fs, parts
if __name__ == "__main__":
    fn = sys.argv[1]; N = int(sys.argv[2]); rng = np.random.default_rng(1)
    pops = [(rng.uniform(.88, 1.15), rng.uniform(.04, .25), rng.uniform(0, .25), rng.uniform(.04, .25), rng.uniform(.2, 1.), float(rng.integers(0, 2))) for _ in range(N)]
    with Pool(18, initializer=init, initargs=(fn,)) as pool:
        res = pool.map(ev, pops)
    good = [(r, p) for r, p in zip(res, pops) if r[0] < .03 and r[1] < .05 and r[2] >= 3]
    print(fn, "success", len(good), "/", N)
    for r, p in sorted(good, key=lambda z: z[0][0])[:5]:
        print([round(v, 3) for v in p], "err %.3f sp %.3f n %d F %.1f" % r[:4], r[4])
    errs = np.array([r[0] for r in res]); print("err percentiles", np.percentile(errs, [5, 25, 50]).round(3))
    Path(fn.replace(".json", "_reseat_good.json")).write_text(json.dumps([list(p) for r, p in good]))
