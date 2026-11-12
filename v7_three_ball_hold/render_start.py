"""Static images of the 2+1 start state (right hand: R in palm, T at fingertips). Output: results/figs/start_*.png"""
import sys
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1] / "v6_humanoid_hands"))
from cascade21 import TwoPlusOneCascade
from juggle import cascade_camera, hide_calibration_sites

out = Path(__file__).parent / "results" / "figs"
c = TwoPlusOneCascade()
m, d = c.model, c.data
for _ in range(180):   # let the fingers close onto the balls (SETTLE_S = 0.2 s)
    c.step()
print("t", d.time)
site = c.hands["right"]["seat"]
p = d.site_xpos[site].copy(); Rm = d.site_xmat[site].reshape(3, 3)
ren = mujoco.Renderer(m, 720, 960)
def shot(name, cam):
    ren.update_scene(d, camera=cam); hide_calibration_sites(ren.scene)
    imageio.imwrite(out / f"start_{name}.png", ren.render())
shot("whole_body", cascade_camera(c.torso_stand_height))
focus = p + Rm @ np.array([0, 0.04, 0.04])
for name, az, el, dist in (("hand_top", 90, -85, 0.30), ("hand_side", 180, -10, 0.30), ("hand_front", 0, -25, 0.30), ("hand_oblique", 135, -35, 0.32)):
    cam = mujoco.MjvCamera(); cam.lookat[:] = focus; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    shot(name, cam)
# also in the palm frame: camera looking along -z from the palm normal side and from the lateral axis
for name, direction, up in (("palm_normal", Rm[:, 2], Rm[:, 1]), ("palm_lateral", Rm[:, 0], Rm[:, 2])):
    pos = focus + direction * 0.28
    v = focus - pos; az = np.degrees(np.arctan2(v[1], v[0])); el = np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))
    cam = mujoco.MjvCamera(); cam.lookat[:] = focus; cam.distance = 0.28; cam.azimuth = az; cam.elevation = el
    shot(name, cam)
for i in (0, 2):
    q = c.balls[i][0]; print("ball", i, "local", (Rm.T @ (d.qpos[q:q+3] - p)).round(3))
