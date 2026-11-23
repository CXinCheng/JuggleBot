"""Paired helps/hurts diagnosis on k=0 observations (dev). Reports per-feature AUC with a
max-AUC permutation null (controls for scanning many features)."""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOUR = ROOT / "v8_robust_control/results/closed_loop_bvp/step2b_four_way_dev_comparison.json"


def auc(x, y):  # y bool; P(x_pos > x_neg) + .5 ties
    p, n = x[y], x[~y]
    if len(p) == 0 or len(n) == 0:
        return 0.5
    g = (p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean()
    return g


def labels(rows_c, rows_z):
    z = {(r["seed"], round(r["jitter_mm"], 4)): r for r in rows_z}
    out = {}
    for r in rows_c:
        k = (r["seed"], round(r["jitter_mm"], 4))
        if r["jitter_mm"] == 0:
            continue
        b = z[k]
        dc = r["catches"] - b["catches"]
        reg = r["is_unsafe"] and not b["is_unsafe"]
        u2s = b["is_unsafe"] and not r["is_unsafe"]
        if reg or (dc < 0 and not u2s):
            out[k] = "hurt"
        elif u2s or (dc > 0 and not reg):
            out[k] = "help"
        else:
            out[k] = "neutral"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", type=Path, required=True)
    ap.add_argument("--cand", default="fixed_bias")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--cond", default="clean")
    a = ap.parse_args()
    obs = json.loads(a.obs.read_text())["rows"]
    if a.cond == "clean":
        four = json.loads(FOUR.read_text())["clean_episodes"]
    else:
        four = json.loads((ROOT / "v8_robust_control/results/closed_loop_bvp/perception_dev_comparison.json").read_text())["episodes"][a.cond]
    lab = labels(four[a.cand], four["zero_base"])
    names, X, keys = None, [], []
    for r in obs:
        k = (r["seed"], round(r["jitter_mm"], 4))
        if k not in lab or lab[k] == "neutral" or not r["obs"]:
            continue
        vec, nm = [], []
        for n, v in r["obs"][0].items():
            f = np.asarray(v).ravel()
            vec += f.tolist(); nm += [f"{n}[{i}]" for i in range(len(f))]
        X.append(vec); keys.append(k); names = nm
    X = np.array(X); y = np.array([lab[k] == "hurt" for k in keys])
    print(f"{a.cand}: help={int((~y).sum())} hurt={int(y.sum())} neutral={sum(v=='neutral' for v in lab.values())}")
    # by jitter
    for j in sorted({k[1] for k in keys}):
        m = np.array([k[1] == j for k in keys])
        print(f"  jitter {j:.2f}: help={int((~y[m]).sum())} hurt={int(y[m].sum())}")
    var = X.std(0) > 1e-9
    idx = np.where(var)[0]
    print("non-constant features:", len(idx), "of", X.shape[1])
    obs_auc = np.array([abs(auc(X[:, i], y) - 0.5) + 0.5 for i in idx])
    rng = np.random.default_rng(0)
    null_max = np.empty(a.perms)
    for t in range(a.perms):
        yp = rng.permutation(y)
        null_max[t] = max(abs(auc(X[:, i], yp) - 0.5) + 0.5 for i in idx[:: max(1, len(idx) // 200)])
    order = np.argsort(-obs_auc)[:10]
    thr95 = np.quantile(null_max, 0.95)
    print(f"permutation null 95% of max-AUC (subsampled features): {thr95:.3f}")
    for o in order:
        print(f"  {names[idx[o]]:<22} AUC={obs_auc[o]:.3f}")
    jit = np.array([k[1] for k in keys])
    print("AUC of jitter level itself (not observable):", round(abs(auc(jit, y) - .5) + .5, 3))
    print("signal found:", bool(obs_auc.max() > thr95))


if __name__ == "__main__":
    main()
