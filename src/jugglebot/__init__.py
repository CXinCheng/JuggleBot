"""Shared JuggleBot modules (from stage v2 onwards).

Copied from v2_single_arm_throw_catch/ (which stays as a frozen snapshot of that stage)
with package-relative imports. Everything here works on a *single-arm* model in the
arm's own base frame: kinematics, ballistic prediction, catch planning (planning),
throw primitives and release-point search (throwing), simulation-based calibration
(calibration), the single-arm simulation harness (simcore) and video helpers (video).
Multi-arm stages plan each arm in its base frame with these modules and execute the
joint trajectories on the arm's joints of the shared scene.
"""


def main() -> None:
    print("Hello from jugglebot!")
