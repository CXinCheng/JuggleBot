"""Record the real v6 right-hand throw (palm pose + finger commands) for replay in the single-hand rig."""
import numpy as np, time
from jugglebot.humanoid.cascade import ContactCascade
c = ContactCascade(n_balls=3)
d, m = c.data, c.model
h = c.hands['right']
rec = {k: [] for k in ('t', 'pos', 'rot', 'ctrl', 'ball0', 'balls_pos', 'balls_vel')}
t0 = time.time()
while d.time < 2.6:
    c.step()
    rec['t'].append(d.time); rec['pos'].append(d.site_xpos[h['seat']].copy())
    rec['rot'].append(d.site_xmat[h['seat']].copy()); rec['ctrl'].append(d.ctrl[h['fingers'].act_indices].copy())
    q = c.balls[0][0]; rec['ball0'].append(d.qpos[q:q+3].copy())
    rec['balls_pos'].append(np.array([d.qpos[q:q+3] for q, _ in c.balls])); rec['balls_vel'].append(np.array([d.qvel[v:v+3] for _, v in c.balls]))
print("wall", time.time()-t0, "events", [(e['kind'], e['side'], round(e.get('time', -1), 3)) for e in c.events[:8]])
rel=np.array([e["time"] for e in c.events if e["kind"]=="release" and e["side"]=="right"]); ev=[(e['kind'], e['side'], float(e.get('time', -1)), int(e.get('ball', -1)) if e.get('ball') is not None else -1) for e in c.events]; print('EVENTS', ev); np.savez("v7_three_ball_hold/results/v6_right_throw.npz", release=rel, **{k: np.array(v) for k, v in rec.items()})
