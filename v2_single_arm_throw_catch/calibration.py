"""Simulation-based calibration of a throw (v2).

The release model of throwing.py (the ball leaves once the specific force stops
pressing it onto the base) is accurate for the translate stroke (~1-2 % speed error),
but the whip presses the ball against the cup wall: it rides up the wall and leaves
30-50 ms late and 7-11 % slow. That error is systematic -- it hardly depends on where
the ball sits on the base -- so it is removed by iterating with the simulator itself as
the model:

    roll out the throw with the ball centred on the base -> measured release (t, p, v)
    -> release velocity that would put the ball on the target point at the target time
    -> v_cmd += (that velocity - measured velocity) -> rebuild the stroke through the same
    release configuration -> repeat

The calibrated plan carries the *measured* release state. What remains afterwards is
the random part (ball position on the base), which the catcher has to absorb.

The rebuilt strokes skip the planner's analytic release checks (rim clearance etc.):
those rest on the biased release prediction, and the rollout checks re-contact for
real. If a correction makes the stroke infeasible (e.g. it grazes a collision limit),
the best rolled-out plan so far is kept when it misses by less than `accept`.
"""

from dataclasses import replace

import numpy as np

from simcore import ArmSim, simulate_throw
from throwing import ThrowPlan, ThrowPlanner


def required_velocity(p: np.ndarray, t: float, p_target: np.ndarray, t_target: float, g: np.ndarray) -> np.ndarray:
    """Ballistic velocity at (p, t) that reaches p_target at t_target."""
    T = t_target - t
    return (p_target - p - 0.5 * g * T ** 2) / T


def calibrate(sim: ArmSim, planner: ThrowPlanner, plan: ThrowPlan, p_target: np.ndarray, t_target: float,
              iters: int = 6, tol: float = 0.002, accept: float = 0.01) -> tuple[ThrowPlan | None, list[float]]:
    """Adjust plan.v_cmd until the simulated ball (centred on the base) passes p_target at t_target
    (t relative to the stroke start). Returns (calibrated plan or None, miss per iteration in m)."""
    g = sim.model.opt.gravity
    history, best = [], None
    for _ in range(iters):
        roll = simulate_throw(sim, plan)
        if roll.outcome != "OK":
            break
        T = t_target - roll.t_release
        at_target = roll.p_release + roll.v_release * T + 0.5 * g * T ** 2
        history.append(float(np.linalg.norm(at_target - p_target)))
        v_req = required_velocity(roll.p_release, roll.t_release, p_target, t_target, g)
        if best is None or history[-1] < best[0]:
            best = (history[-1], replace(plan, t_release=roll.t_release, p_release=roll.p_release,
                                         v_release=roll.v_release, v_target=v_req, calibrated=True))
        if history[-1] < tol:
            break
        v_cmd = plan.v_cmd + (v_req - roll.v_release)
        new = planner.build(plan.mode, plan.q_release, v_cmd, plan.axis, lambda p: v_req, check_release=False)
        if new is None:
            break  # the corrected stroke violates a constraint
        plan = new
    if best is not None and best[0] < accept:
        return best[1], history
    return None, history
