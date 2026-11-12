"""Record the 2+1-start cascade to MP4. Example: uv run python v7_three_ball_hold/record_cascade21.py --throws 8 --speed 0.5"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1] / "v6_humanoid_hands"))
from cascade21 import TwoPlusOneCascade
from juggle import cascade_camera, hide_calibration_sites
from jugglebot.video import VideoRecorder
ap = argparse.ArgumentParser(); ap.add_argument("--throws", type=int, default=8); ap.add_argument("--speed", type=float, default=0.5)
ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "cascade21_start.mp4")); ap.add_argument("--timeout", type=float, default=12.0)
a = ap.parse_args()
c = TwoPlusOneCascade()
rec = VideoRecorder(c.model, a.out, fps=60, speed=a.speed, camera=cascade_camera(c.torso_stand_height))
try:
    while c.data.time < a.timeout and c.throw_count < a.throws and c.failure_event is None:
        c.step(); rec.capture(c.data, scene_callback=hide_calibration_sites)
    for _ in range(600):
        c.step(); rec.capture(c.data, scene_callback=hide_calibration_sites)
finally:
    rec.close()
print("frames", rec.frames, "throws", c.throw_count, "failure", c.failure_event)
