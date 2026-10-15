"""UR5e + LEAP hand modules (v4's, shared from v5 on).

v4_hand_grasp/ stays a frozen snapshot of that stage; these are its modules made side-aware
(right / left LEAP hand, the left one the mirror image of the right) and extended to several arms:

    model      single-arm and multi-arm model assembly, the right <-> left palm-frame mirror map
    handsim    single-arm harness (planning model: dynamics, rollouts), LineStroke, bowl sites
    world      multi-arm simulation: Arm views, ball contacts per arm
    kinematics, prediction, planning   v4's IK / FK, ballistic fit (with prior), catch planning
               (incl. the "direct" strategy of E5)
    throwing   postures, the bowl-hand finger schedule, whip strokes, throw calibration
"""
