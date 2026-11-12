"""Images of the ring-finger(+thumb) palm hold + fingertip hold of the last ball (results/pinky_{A,B}.json)."""
import json, sys
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
sys.path.insert(0, str(Path(__file__).parent))
from search_hold2 import Problem
P = Problem("right", 10.0); rig = P.rig; out = Path(__file__).parent / "results" / "figs"
ren = mujoco.Renderer(rig.m, 720, 960)
for v in "AB":
    x = np.array(json.loads((Path(__file__).parent / "results" / f"pinky_{v}.json").read_text())["x"])
    P.simulate(x, T=0.6); p, Rm = rig.seat_frame(); focus = p + Rm @ np.array([0, 0.04, 0.04])
    for name, az, el in (("oblique", 135, -35), ("top", 90, -85), ("side", 180, -10)):
        cam = mujoco.MjvCamera(); cam.lookat[:] = focus; cam.distance = 0.32; cam.azimuth = az; cam.elevation = el
        ren.update_scene(rig.d, camera=cam)
        for g in ren.scene.geoms[:ren.scene.ngeom]:
            if g.objtype == mujoco.mjtObj.mjOBJ_SITE: g.rgba[3] = 0
        imageio.imwrite(out / f"pinky_{v}_{name}.png", ren.render())
