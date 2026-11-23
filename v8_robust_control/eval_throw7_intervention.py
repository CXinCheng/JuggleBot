"""Step 2: Paired Evaluation of Throw 7 Collision Avoidance Interventions.

Tests on:
- Seed 440000 (Baseline original calibration)
- Seed 440002 (Modified with flight 4 lateral zeroed)

Candidates:
1. Baseline: zero delay, zero throw residual
2. Deferral candidates: defer_s in [0.005, 0.010, 0.015, 0.020, 0.025] s
3. Vertical velocity offsets: dvz in [-0.08, -0.04, +0.04, +0.08] m/s

Metrics recorded:
- Predicted clearance at t=2.220s decision
- Actual release timestamp of Ball 2 (t_rel_act)
- Actual release velocity of Ball 2
- Actual minimum ball-to-ball distance and clearance (Ball 0 vs Ball 2)
- Posture/apex check: max ball height
- Reception margin for Flight 7 (left hand ready time vs ball crossing)
- Confirmed catches, releases, elapsed time, failure event.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import mujoco

from v8_robust_control.task_env import TaskConfig
from v8_robust_control.closed_loop_bvp import ClosedLoopResidualEnv, ClosedLoopConfig
from v8_robust_control.eval_throw_ablation import AblatedThrowPolicy, REF_PATH
from v8_robust_control.task_control import TaskCommand, decode_action, THROW_RESIDUAL_LIMITS

BALL_RADIUS = 0.0425
BALL_DIAMETER = 2 * BALL_RADIUS


def run_intervention(seed: int, cancel_f4: bool, defer_s: float = 0.0, dvz: float = 0.0):
    cfg = ClosedLoopConfig()
    env = ClosedLoopResidualEnv(
        TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=0.0, noise_std_m=0.001),
        reference=str(REF_PATH), closed_loop_cfg=cfg, residual_scale=0.0)
    env.rule_policy = AblatedThrowPolicy('zero', cfg)
    env.reset(seed=seed)
    c = env.controller
    m = c.model
    d = c.data

    if cancel_f4:
        c._nominal_inputs['return:right:5'][1] = 0.0

    step_records = []
    contacts = []
    phys_releases = []
    throw_decision_info = {}

    orig_step = c.step
    def step_hook():
        orig_step()
        t = float(d.time)

        # Track balls 0 and 2
        q0, _ = c.balls[0]
        q2, _ = c.balls[2]
        p0 = d.qpos[q0:q0+3].copy()
        p2 = d.qpos[q2:q2+3].copy()
        dist_02 = float(np.linalg.norm(p0 - p2))

        step_records.append((t, p0, p2, dist_02))

        for i in range(d.ncon):
            ct = d.contact[i]
            g1, g2 = ct.geom1, ct.geom2
            b0_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'ball0')
            b2_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'ball2')
            if (g1 == b0_geom and g2 == b2_geom) or (g1 == b2_geom and g2 == b0_geom):
                contacts.append((t, 'ball0-ball2', float(ct.dist)))

    c.step = step_hook

    # Hook decision callback
    orig_command = c._command
    def command_hook(request):
        if request.kind == 'throw' and request.side == 'right' and request.ball_id == 2 and request.released == 6:
            # This is Decision 7!
            nom = np.array(request.nominal_parameters)
            params = nom.copy()
            if abs(dvz) > 1e-6:
                params[2] += dvz

            # Only defer on the FIRST attempt (attempt == 0)
            actual_defer = defer_s if request.attempt == 0 else 0.0

            cmd = TaskCommand(tuple(params), defer_s=actual_defer)
            c.decisions.append(dict(
                kind=request.kind, side=request.side, ball_id=request.ball_id,
                time_s=request.time_s, released=request.released, attempt=request.attempt,
                nominal_parameters=list(request.nominal_parameters),
                parameters=tuple(params), defer_s=actual_defer, tilt_scale=1.0
            ))
            throw_decision_info['t_dec'] = float(request.time_s)
            throw_decision_info['defer_s'] = defer_s
            throw_decision_info['dvz'] = dvz
            return cmd, params

        return orig_command(request)

    c._command = command_hook

    done = trunc = False
    k = 0
    while not (done or trunc):
        _, _, done, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        k += 1

    fe = info.get('failure_event')
    catches = int(info.get('catches', 0))
    releases = int(info.get('releases', 0))
    elapsed_s = float(info.get('elapsed_s', 0.0))
    completed_ids = info.get('completed_ids', [])
    caught_events = sorted(c.caught_events)

    if hasattr(c, 'events'):
        for e in c.events:
            if e.get('kind') == 'release':
                phys_releases.append(dict(
                    time=round(float(e['time']), 4),
                    side=e['side'],
                    ball=int(e['ball']),
                    flight=e.get('flight'),
                    vel=[round(float(x), 4) for x in e['velocity']],
                    pos=[round(float(x), 4) for x in e.get('position', [])] if 'position' in e else None,
                ))

    # Min distance between Ball 0 and Ball 2 after t=2.30
    post_col_steps = [s for s in step_records if s[0] >= 2.30]
    min_dist_02 = None
    min_t_02 = None
    if post_col_steps:
        best = min(post_col_steps, key=lambda s: s[3])
        min_t_02 = best[0]
        min_dist_02 = best[3]

    # Ball 2 release
    b2_rel = [r for r in phys_releases if r['ball'] == 2 and r['time'] > 2.2]
    last_b2_rel = b2_rel[-1] if b2_rel else None

    # Peak height of Ball 2
    b2_steps = [s for s in step_records if s[0] >= 2.30]
    peak_b2_z = max([s[2][2] for s in b2_steps]) if b2_steps else None

    env.close()

    return {
        'seed': seed,
        'cancel_f4': cancel_f4,
        'defer_s': defer_s,
        'dvz': dvz,
        'catches': catches,
        'releases': releases,
        'elapsed_s': elapsed_s,
        'completed_ids': completed_ids,
        'caught_events': caught_events,
        'failure_event': fe,
        'min_dist_02_mm': (min_dist_02 * 1000) if min_dist_02 else None,
        'clearance_mm': ((min_dist_02 - BALL_DIAMETER) * 1000) if min_dist_02 else None,
        'min_t_02': min_t_02,
        'b2_release': last_b2_rel,
        'peak_b2_z': peak_b2_z,
        'ball_collision_contacts': len(contacts),
    }


def main():
    print("==========================================================================")
    print("STEP 2: SINGLE-VARIABLE PAIRED EXPERIMENT ON THROW 7 COLLISION AVOIDANCE")
    print("==========================================================================\n")

    # We evaluate both branches:
    # Branch 440000 (Baseline)
    # Branch 440002 (Flight 4 Cancelled)

    delay_candidates = [0.0, 0.005, 0.010, 0.015, 0.020]
    results = []

    for seed, cancel_f4, name in [
        (440000, False, "Seed 440000 (Baseline)"),
        (440002, True, "Seed 440002 (Flight 4 Lateral Zeroed)"),
    ]:
        print(f"\n==================== {name} ====================")
        for d_s in delay_candidates:
            res = run_intervention(seed, cancel_f4, defer_s=d_s, dvz=0.0)
            results.append(res)
            cl_str = f"{res['clearance_mm']:+.1f} mm" if res['clearance_mm'] is not None else "N/A"
            dist_str = f"{res['min_dist_02_mm']:.1f} mm" if res['min_dist_02_mm'] is not None else "N/A"
            b2_t = f"{res['b2_release']['time']:.4f}s" if res['b2_release'] else "None"
            print(f"Delay = {d_s*1000:4.1f} ms | Catches: {res['catches']:2d}/30 | Releases: {res['releases']:2d} | Elapsed: {res['elapsed_s']:.3f}s | B2 Rel: {b2_t} | MinDist: {dist_str} (Clr: {cl_str}) | Failure: {res['failure_event']}")

    # Check if vertical speed offset is needed
    print("\n---------------- Testing Vertical Speed Offsets (dvz) ----------------")
    for seed, cancel_f4, name in [
        (440000, False, "Seed 440000 (Baseline)"),
        (440002, True, "Seed 440002 (Flight 4 Lateral Zeroed)"),
    ]:
        print(f"\n>>> {name}:")
        for dvz in [-0.06, -0.04, +0.04, +0.06]:
            res = run_intervention(seed, cancel_f4, defer_s=0.0, dvz=dvz)
            results.append(res)
            cl_str = f"{res['clearance_mm']:+.1f} mm" if res['clearance_mm'] is not None else "N/A"
            dist_str = f"{res['min_dist_02_mm']:.1f} mm" if res['min_dist_02_mm'] is not None else "N/A"
            b2_t = f"{res['b2_release']['time']:.4f}s" if res['b2_release'] else "None"
            print(f"dvz = {dvz:+.2f} m/s | Catches: {res['catches']:2d}/30 | Releases: {res['releases']:2d} | Elapsed: {res['elapsed_s']:.3f}s | B2 Rel: {b2_t} | MinDist: {dist_str} (Clr: {cl_str}) | Failure: {res['failure_event']}")

    out_file = Path("v8_robust_control/results/closed_loop_bvp/throw7_intervention_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {out_file}")


if __name__ == '__main__':
    main()
