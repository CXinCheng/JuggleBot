"""Re-run the 30-throw 2+1 cascade and dump the diagnostics of the failing reception plan (NO_FEASIBLE_RECEPTION)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cascade21 import TwoPlusOneCascade
from jugglebot.humanoid.cascade import ContactCascade

args = [a for a in sys.argv[1:] if not a.startswith("--")]
throws = int(args[0]) if args else 30
out = Path(args[1]) if len(args) > 1 else Path(__file__).parent / "results" / "diag_30.json"
V6 = "--v6" in sys.argv
c = ContactCascade(n_balls=3) if V6 else TwoPlusOneCascade()
t0 = time.time()
res = c.run_continuous(throws=throws, timeout=40.0, guard=True)
print("wall %.1fs success %s error %s time %.3f throws %d" % (time.time() - t0, res.get("success"), res.get("error"), c.data.time, c.throw_count), flush=True)
plans = [e for e in c.events if e["kind"] == "return_plan"]
fails = [e for e in plans if not all(e.get("diagnostics", {}).get(k, False) for k in ("captured", "collision_free", "ready_for_next_throw"))]
print("failing return_plans:", len(fails))
for e in fails:
    print(json.dumps(e, default=float)[:2500])
out.write_text(json.dumps(dict(failure=c.failure_event, last_plans=plans[-4:], events_tail=c.events[-12:], events=c.events), default=float, indent=1))
