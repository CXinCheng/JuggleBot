"""Noise robustness of the 2+1 cascade / ending (v6-style partial reception-position noise, std in mm, reproducible seeds).

  cascade: uv run python v7_three_ball_hold/noise21.py cascade --noise-mm 0 1 3 5 --seeds 0 1 2 --throws 30 --jobs 6
  ending:  uv run python v7_three_ball_hold/noise21.py ending  --noise-mm 0 1 3 5 --seeds 0 1 2 --stop-after 8 --jobs 6
One JSON line per trial is appended to --output (default results/noise21_<mode>.jsonl); completed trials are skipped on restart.
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def trial(case):
    mode, noise, seed, n = case
    from cascade21 import TwoPlusOneCascade
    from end21 import EndingCascade
    t0 = time.time()
    if mode == "cascade":
        c = TwoPlusOneCascade(obs_noise=noise / 1000.0, seed=seed)
        r = c.run_continuous(throws=n, timeout=40.0, guard=True)
        return dict(mode=mode, noise_mm=noise, seed=seed, requested=n, success=bool(r["success"]), error=r.get("error"), releases=c.throw_count,
                    validated=r.get("validated_flights"), sim_time=float(c.data.time), drops=r.get("drops"), ball_contacts=r.get("ball_contacts"),
                    wall_s=time.time() - t0)
    c = EndingCascade(stop_after=n, obs_noise=noise / 1000.0, seed=seed)
    err = None
    try:
        while c.data.time < 40.0 and c.end_result is None:
            c.step()
    except RuntimeError as e:
        err = str(e)
    ok = bool(c.end_result and c.end_result["ok"]) and err is None
    return dict(mode=mode, noise_mm=noise, seed=seed, stop_after=n, success=ok, error=err, releases=c.throw_count, sim_time=float(c.data.time),
                end_result=c.end_result, wall_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cascade", "ending"))
    ap.add_argument("--noise-mm", type=float, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--throws", type=int, default=30)
    ap.add_argument("--stop-after", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    out = Path(a.output or Path(__file__).parent / "results" / f"noise21_{a.mode}.jsonl")
    n = a.throws if a.mode == "cascade" else a.stop_after
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["noise_mm"], r["seed"]))
    cases = [(a.mode, nz, sd, n) for nz in a.noise_mm for sd in a.seeds if (nz, sd) not in done]
    with Pool(a.jobs) as pool, out.open("a") as f:
        for r in pool.imap_unordered(trial, cases):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(f"noise {r['noise_mm']:g} mm seed {r['seed']}: success={r['success']} releases={r['releases']} error={r['error']} wall={r['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
