"""Run the whole-body 2+1 start and ending. Example: uv run python v7_three_ball_hold/run_end21.py --stop-after 4"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from end21 import EndingCascade, CATCH

ap = argparse.ArgumentParser()
ap.add_argument("--stop-after", type=int, default=4)
ap.add_argument("--timeout", type=float, default=8.0)
ap.add_argument("--catch", default=str(CATCH))
ap.add_argument("--lat", type=float, default=None)
ap.add_argument("--elbow", type=float, default=None)
ap.add_argument("--video", default=None)
ap.add_argument("--speed", type=float, default=0.5)
a = ap.parse_args()
c = EndingCascade(stop_after=a.stop_after, catch_file=a.catch)
if a.elbow is not None:
    c.ELBOW_OUT_END_M = a.elbow
if a.lat is not None:
    c.LATERAL_RATIO = a.lat
rec = None
if a.video:
    sys.path.insert(0, str(Path(__file__).parents[1] / "v6_humanoid_hands"))
    from juggle import cascade_camera, hide_calibration_sites
    from jugglebot.video import VideoRecorder
    rec = VideoRecorder(c.model, a.video, fps=60, speed=a.speed, camera=cascade_camera(c.torso_stand_height))
t0 = time.time()
err = None
try:
    while c.data.time < a.timeout and c.end_result is None:
        c.step()
        if rec:
            rec.capture(c.data, scene_callback=hide_calibration_sites)
except RuntimeError as e:
    err = str(e)
if rec:
    rec.close()
print("wall %.1fs sim %.3f throws %d error %s" % (time.time() - t0, c.data.time, c.throw_count, err))
if c.end:
    for row in c.end.get("trace", []):
        print("trace", row)
print("end_result", json.dumps(c.end_result, indent=1) if c.end_result else None)
for e in c.events:
    if e["kind"] in ("end_start", "release", "catch") and e.get("time", 0) > 0:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k in ("time", "kind", "side", "ball", "count", "impact", "goal", "relspeed", "catch_p", "p")})
