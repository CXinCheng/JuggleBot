"""Recalibrate the v6 optimizer seeds on the 2+1 timeline (fixed-point iteration, like v6 calibrate.py but starting from the 2+1 start).

Each iteration runs the 2+1 cascade for --throws flights with the current seeds, then caches, for every throw / reception, the command
that was finally accepted (score < 0.02 for throws, score 0 for receptions) as that throw count's first-attempt guess. Seeds only
change for throws that were actually reached. The loop ends at the first run that reaches the target: its input seeds are validated by that very run (the simulation is
deterministic) and are written out; otherwise it stops after --max-iter iterations. Result: results/seeds_2plus1.json (used automatically by
TwoPlusOneCascade). The two hand-calibrated seeds throw:right:0 / throw:right:2 are never overwritten.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from cascade21 import TwoPlusOneCascade, SEEDS_FILE

KEEP = ("throw:right:0", "throw:right:2")


def collect(cascade, seeds):
    execution = dict(seeds.get("execution", {}))
    attempts = {}
    count = 0
    for e in cascade.events:
        if e["kind"] == "release":
            count = e["count"]
        elif e["kind"] in ("plan", "plan_rejected"):
            attempts.setdefault(f"throw:{e['side']}:{count}", []).append(e)
        elif e["kind"] == "return_plan":
            if e.get("score", float(any(e["error"]))) == 0.0:
                execution[f"return:{e['side']}:{count}"] = list(e["parameters"])
    stats = {}
    for key, plans in attempts.items():
        if key in KEEP:
            continue
        last = plans[-1]
        stats[key] = (len(plans), plans[0].get("score", -1.0), last.get("score", -1.0))
        if last.get("score", 1e9) < 0.02 and "command" in last:
            execution[key] = list(last["command"])
    return execution, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--throws", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--out", default=str(SEEDS_FILE))
    ap.add_argument("--fresh", action="store_true", help="ignore an existing seeds file")
    a = ap.parse_args()
    out = Path(a.out)
    seeds = json.loads(out.read_text()) if out.exists() and not a.fresh else {"execution": {}}
    for it in range(a.max_iter):
        t0 = time.time()
        c = TwoPlusOneCascade(seeds=seeds, seeds_file=None)
        res = c.run_continuous(throws=a.throws, timeout=a.timeout, guard=True)
        execution, stats = collect(c, seeds)
        changed = [k for k in execution if seeds.get("execution", {}).get(k) is None or not np.allclose(execution[k], seeds["execution"][k], atol=1e-9)]
        retries = {k: v for k, v in stats.items() if v[0] > 1}
        print(f"[iter {it}] wall {time.time()-t0:.0f}s success={res['success']} error={res.get('error')} throws={c.throw_count} "
              f"validated={res.get('validated_flights')} changed_seeds={len(changed)} retried_throws={len(retries)}", flush=True)
        for k, v in sorted(retries.items()):
            print(f"    retry {k}: attempts={v[0]} first_score={v[1]:.3f} final={v[2]:.3f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / f"{out.stem}_iter{it}_input.json").write_text(json.dumps(seeds, indent=1) + "\n")
        if res["success"]:
            # the INPUT seeds of a successful run are validated by that very run (deterministic simulation): accept them as they are
            out.write_text(json.dumps(seeds, indent=1) + "\n")
            print(f"ACCEPTED: the seeds used by iteration {it} reproduce a successful {a.throws}-flight run from scratch", flush=True)
            return
        seeds = dict(seeds, execution=execution)
    print("NOT ACCEPTED within max-iter", flush=True)


if __name__ == "__main__":
    main()
