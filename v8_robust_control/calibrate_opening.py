"""Finite opening calibration and phase-limited policy evaluation under contact physics."""
import argparse
from dataclasses import asdict, dataclass
from itertools import combinations, product
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from jugglebot.humanoid import cascade as C
from jugglebot.humanoid.kinematics import HumanoidArmKinematics

from .compare_residual import digest
from .task_corrections import run_jobs, load_corrections, load_parent
from .task_env import TaskConfig, TaskResidualEnv, _EpisodeCascade, runtime_sources
from .task_control import TaskCommand, RecordedReceptionRejected, validate_command
from .train_task import TaskCurriculum, write_new


@dataclass(frozen=True)
class OpeningProfile:
    settle_s: float = 0.
    forward_up_m_s: float = 0.
    velocity_bias_m_s: tuple[float,float,float] = (0.,0.,0.)
    launch_velocity_m_s: tuple = ((0.,0.,0.),)*3
    reception_offset_m: tuple[float,float,float] = (0.,0.,0.)
    release_lead_s: float = 0.
    reception_start_release: int = 0
    reception_planner: str = 'off'
    uncaught_braking: bool = False

    def __post_init__(self):
        velocity = np.asarray(self.velocity_bias_m_s,dtype=float)
        launches = np.asarray(self.launch_velocity_m_s,dtype=float)
        reception = np.asarray(self.reception_offset_m,dtype=float)
        if (not np.isfinite([self.settle_s,self.forward_up_m_s]).all()
                or not 0. <= self.settle_s <= .25 or not 0. <= self.forward_up_m_s <= .2
                or velocity.shape!=(3,) or not np.isfinite(velocity).all() or np.any(np.abs(velocity)>.2)
                or launches.shape!=(3,3) or not np.isfinite(launches).all() or np.any(np.abs(launches)>.2)
                or reception.shape!=(3,) or not np.isfinite(reception).all() or np.any(np.abs(reception)>.02)
                or not np.isfinite(self.release_lead_s) or abs(self.release_lead_s)>.003
                or type(self.reception_start_release) is not int or not 0<=self.reception_start_release<=6
                or self.reception_planner not in ('off','audit','project')
                or type(self.uncaught_braking) is not bool):
            raise ValueError('Require bounded opening settings: settle<=250 ms, velocity<=0.2 m/s, reception<=20 mm, lead<=3 ms')
        object.__setattr__(self,'velocity_bias_m_s',tuple(velocity.tolist()))
        object.__setattr__(self,'launch_velocity_m_s',tuple(tuple(row) for row in launches.tolist()))
        object.__setattr__(self,'reception_offset_m',tuple(reception.tolist()))


def joint_profiles():
    profiles = [OpeningProfile()]
    for lane,catch_x in product((-.01,0.,.01),(0.,.005,.01)):
        if lane==0. and catch_x==0.:
            continue
        profiles.append(OpeningProfile(launch_velocity_m_s=((lane,0.,0.),(-lane,0.,0.),(lane,0.,0.)),
                                       reception_offset_m=(catch_x,0.,0.)))
    for lead in (-.001,.001):
        profiles.append(OpeningProfile(release_lead_s=lead,reception_offset_m=(.005,0.,0.)))
    for catch_x in (.005,.01):
        profiles.append(OpeningProfile(launch_velocity_m_s=((0.,0.,-.01),)*3,
                                       reception_offset_m=(catch_x,0.,0.)))
    return [asdict(p) for p in profiles]


def retained_height_profiles():
    return [asdict(OpeningProfile())]+[asdict(OpeningProfile(
        launch_velocity_m_s=((0.,0.,0.),(0.,0.,0.),(0.,0.,velocity)),
        reception_offset_m=(0.,0.,height),reception_start_release=4))
        for velocity,height in product((-.05,-.10,-.15),(0.,-.005,.005))]


class ClearanceCascade(_EpisodeCascade):
    """Read-only measurements after normal physical steps, including failed steps."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._clearance_geoms = tuple(sorted(set(g for values in self._wrist_geoms.values() for g in values)
            |{g for g in range(self.model.ngeom)
              if self.model.body(self.model.geom_bodyid[g]).name in ('lh_palm','rh_palm')
              and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])}))
        # Cache immutable geometry classification; contact scans run at 1 kHz.
        self._clearance_body_names = tuple(self.model.body(self.model.geom_bodyid[g]).name for g in range(self.model.ngeom))
        self._clearance_tags = tuple(-1 if n.startswith('ball') else 1 if n.startswith('rh_')
                                     else 2 if n.startswith('lh_') else 0 for n in self._clearance_body_names)
        self._clearance_elbows = tuple((self.model.body(s+'_elbow_link').id,
                                       self.model.body(s+'_shoulder_pitch_link').id) for s in self.hands)
        self.clearance = dict(min_gap_m=None, pair=None, time_s=None, samples=0, peak_ball_height_m=0.,
            min_hand_stand_gap_m=None,hand_stand_time_s=None,hand_stand_samples=0,
            max_robot_penetration_m=0.,robot_contact_pair=None,min_elbow_margin_m=None)

    def step(self):
        try:
            return super().step()
        finally:
            if hasattr(self,'clearance') and not self.silent:
                self.measure_robot_clearance()
                self.measure_clearance()

    def measure_robot_clearance(self):
        c,m,d = self.clearance,self.model,self.data
        for contact in d.contact:
            if contact.dist>=0.:
                continue
            a,b = self._clearance_tags[contact.geom1],self._clearance_tags[contact.geom2]
            if a==-1 or b==-1 or (a>0 and a==b):
                continue
            penetration = max(0.,-float(contact.dist))
            if penetration>c['max_robot_penetration_m']:
                c.update(max_robot_penetration_m=penetration,
                         robot_contact_pair=[m.geom(g).name for g in (contact.geom1,contact.geom2)])
        for elbow,shoulder in self._clearance_elbows:
            margin = .10-(abs(d.xpos[elbow,1])-abs(d.xpos[shoulder,1]))
            if c['min_elbow_margin_m'] is None or margin<c['min_elbow_margin_m']:
                c['min_elbow_margin_m'] = float(margin)
        # Proximity is a capped, 10 ms diagnostic; native safety still runs each step.
        if round(d.time/m.opt.timestep)%10==0:
            for geom in self._clearance_geoms:
                gap = float(mujoco.mj_geomDistance(m,d,self._stand_geom,geom,.05,None))
                c['hand_stand_samples'] += 1
                if c['min_hand_stand_gap_m'] is None or gap<c['min_hand_stand_gap_m']:
                    c.update(min_hand_stand_gap_m=gap,hand_stand_time_s=float(d.time))

    def measure_clearance(self):
        c = self.clearance
        c['peak_ball_height_m'] = max(c['peak_ball_height_m'],
                                     max(float(self.data.qpos[q+2]) for q,_ in self.balls))
        if len(self.airborne)<2:
            return
        free = [i for i in self.airborne if not any(self.ball_contact(i,s) for s in self.hands)]
        for i,j in combinations(free,2):
            x = self.data.qpos[self.balls[i][0]:self.balls[i][0]+3]
            y = self.data.qpos[self.balls[j][0]:self.balls[j][0]+3]
            radius = sum(self.model.geom_size[self.ball_geom_ids[k],0] for k in (i,j))
            gap = float(np.linalg.norm(x-y)-radius)
            c['samples'] += 1
            if c['min_gap_m'] is None or gap<c['min_gap_m']:
                c.update(min_gap_m=gap,pair=[i,j],time_s=float(self.data.time))


class ReceptionPlanningCascade(ClearanceCascade):
    """Bounded kinematic reception preview; all acceptance remains physical."""
    def __init__(self, reception_planner='audit', **kwargs):
        super().__init__(**kwargs)
        self.reception_planner = reception_planner
        self.reception_plans = []

    def preview_reception(self, side, reference):
        path,stop,axis = reference
        hand,m = self.hands[side],self.model
        kin = HumanoidArmKinematics(m,side,tcp_site=m.site(hand['seat']).name)
        mujoco.mj_copyData(kin.data,m,self.data)
        q = self.data.qpos[kin.qpos_adr].copy()
        times = sorted(set([path.t0,.5*(path.t0+stop.t0),stop.t0,
                            stop.t0+.5*stop.duration,stop.t1]))
        samples,score = [],0.
        for t in times:
            position = (path if t<=stop.t0 else stop).sample(t)[0]
            normal = path.orientation(t)[0][:,2] if t<=stop.t0 else axis
            solved = kin.solve(position,normal,q,q_nominal=hand['q'] if t<=stop.t0 else hand['catch_q'],max_iters=60)
            q = solved.q
            kin.data.qpos[kin.qpos_adr] = q
            mujoco.mj_kinematics(m,kin.data)
            mujoco.mj_comPos(m,kin.data)
            mujoco.mj_collision(m,kin.data)
            shoulder,elbow = (m.body(side+n).id for n in ('_shoulder_pitch_link','_elbow_link'))
            outward = abs(kin.data.xpos[elbow,1])-abs(kin.data.xpos[shoulder,1])
            clearance = kin.data.xpos[shoulder,2]-kin.data.xpos[elbow,2]
            gap = min(float(mujoco.mj_geomDistance(m,kin.data,self._stand_geom,g,.05,None)) for g in self._clearance_geoms)
            penetration = 0.
            for contact in kin.data.contact:
                a,b = self._clearance_tags[contact.geom1],self._clearance_tags[contact.geom2]
                if a==-1 or b==-1 or (a>0 and a==b):
                    continue
                penetration = max(penetration,-float(contact.dist))
            violations = [max(0.,solved.pos_err-.003)/.003,max(0.,solved.axis_err-.05)/.05,
                max(0.,outward-C.MAX_ELBOW_OUT_M)/.01,
                max(0.,C.MIN_SHOULDER_ELBOW_CLEARANCE_M-clearance)/.01,
                max(0.,abs(q[5])-C.MAX_WRIST_PITCH_RAD)/.05,
                max(0.,penetration-C.MAX_SELF_PENETRATION_M)/.004,max(0.,.003-gap)/.003]
            value = float(sum(violations))
            score += value
            samples.append(dict(time_s=float(t),phase='approach' if t<=stop.t0 else 'uncaught_cushion',
                position_error_m=solved.pos_err,axis_error_rad=solved.axis_err,
                elbow_out_m=float(outward),shoulder_elbow_clearance_m=float(clearance),wrist_pitch_rad=float(q[5]),
                hand_stand_gap_m=gap,robot_penetration_m=penetration,violation=value))
        # Successful capture switches references before the uncaught tail ends.
        # Keep its feasibility distinct from that of reaching the impact point.
        return dict(feasible=score<=1e-9,violation=score,
            approach_feasible=all(s['violation']<=1e-9 for s in samples if s['phase']=='approach'),
            uncaught_cushion_feasible=all(s['violation']<=1e-9 for s in samples if s['phase']=='uncaught_cushion'),
            samples=samples)

    def plan_reception(self, side):
        hand = self.hands[side]
        request = self._request('reception',side,hand['incoming'])
        try:
            command,parameters = self._command(request)
        except RecordedReceptionRejected:
            self.failure_event = dict(reason='NO_FEASIBLE_RECEPTION',time=float(self.data.time),side=side,ball=hand['incoming'])
            raise
        started = time.perf_counter()
        nominal = self.reception_reference(side,parameters,tilt_scale=command.tilt_scale)
        baseline = self.preview_reception(side,nominal)
        chosen,chosen_parameters = nominal,parameters
        candidates = [dict(command=dict(parameters=parameters.tolist(),tilt_scale=command.tilt_scale),preview=baseline)]
        if self.reception_planner=='project' and not baseline['feasible']:
            lower,upper = self.reception_bounds(side)
            deltas = [(x,0.,z,0.,0.) for x,z in ((.005,0.),(.01,0.),(0.,.005),(0.,.01),(.005,.005))]
            deltas += [(0.,0.,0.,x,0.) for x in (-.05,.05)]
            deltas += [(0.,0.,0.,0.,x) for x in (-.05,.05)]
            feasible = []
            for delta in deltas:
                p = np.clip(parameters+delta,lower,upper)
                c = TaskCommand(tuple(p),tilt_scale=command.tilt_scale)
                try:
                    validate_command(request,c,(lower,upper))
                except ValueError:
                    continue
                reference = self.reception_reference(side,p,tilt_scale=c.tilt_scale)
                preview = self.preview_reception(side,reference)
                candidates.append(dict(command=dict(parameters=p.tolist(),tilt_scale=c.tilt_scale),preview=preview))
                if preview['feasible']:
                    feasible.append((float(np.linalg.norm(np.asarray(delta)/[.01,.01,.01,.1,.1])),reference,p))
            if feasible:
                _,chosen,chosen_parameters = min(feasible,key=lambda row:row[0])
        self.apply_reception_reference(hand,*chosen)
        hand['return_parameters'] = chosen_parameters
        self.reception_plans.append(dict(side=side,ball=request.ball_id,released=request.released,time_s=request.time_s,
            mode=self.reception_planner,baseline_feasible=baseline['feasible'],
            changed=bool(np.max(np.abs(chosen_parameters-parameters))>1e-12),
            fallback=not baseline['feasible'] and not any(c['preview']['feasible'] for c in candidates),
            selected_parameters=chosen_parameters.tolist(),wall_s=time.perf_counter()-started,candidates=candidates))
        # A preview cannot certify capture. If no candidate is feasible, execute
        # the original reference and let native physics report the actual outcome.


class OpeningEnv(TaskResidualEnv):
    def __init__(self, config, reference, profile):
        super().__init__(config,reference)
        self._template = (ClearanceCascade(throws=config.throws,uncaught_braking=profile.uncaught_braking,
                                           estimator=self.estimator)
                          if profile.reception_planner=='off' else
                          ReceptionPlanningCascade(throws=config.throws,reception_planner=profile.reception_planner,
                                                   uncaught_braking=profile.uncaught_braking,estimator=self.estimator))
        if self._template.data.time!=0.:
            raise ValueError('Schedule calibration requires an unstepped template')
        for hand in self._template.hands.values():
            hand['throw_time'] += profile.settle_s


def reception_residual(obs, proposal, cap_m=.002):
    """Measured event context only: after two releases, reception position only."""
    if not np.isfinite(cap_m) or not 0. <= cap_m <= .06:
        raise ValueError('Reception cap must stay inside the declared 60 mm bounds')
    proposal = np.asarray(proposal,dtype=float)
    if proposal.shape!=(6,) or not np.isfinite(proposal).all() or np.any(np.abs(proposal)>1.):
        raise ValueError('Require an already bounded policy proposal')
    context = obs['context']
    output = np.zeros(6)
    if context[1]>.5 and round(float(context[5])*30)>=2:
        output[:3] = proposal[:3]*(cap_m/.06)
    return output


def opening_action(obs, profile, probe):
    output = np.zeros(6)
    released = round(float(obs['context'][5])*30)
    if obs['context'][0]>.5 and released<3:
        velocity = np.asarray(profile.launch_velocity_m_s[released]).copy()
        if released==0:
            velocity += (np.asarray(probe,dtype=float)+profile.velocity_bias_m_s
                         +np.array([profile.forward_up_m_s,0.,profile.forward_up_m_s]))
        output[:3],output[4] = velocity/[.75,.75,1.25],profile.release_lead_s/.015
    elif obs['context'][1]>.5 and profile.reception_start_release<=released<6:
        output[:3] = np.asarray(profile.reception_offset_m)/.06
    if np.any(np.abs(output)>1.):
        raise ValueError('Opening calibration exceeds the existing action bounds')
    return output


def trial(job):
    reference, profile_dict, seed, jitter, probe, goal, candidate, cap = job
    torch.set_num_threads(1)
    profile = OpeningProfile(**profile_dict)
    # Nominal profile calibration must execute its declared command even at
    # zero innovation. The legacy policy retains its original measured reference.
    base = OpeningEnv(TaskConfig(),reference,profile)
    if candidate is None:
        # Disable only the innovation gate for fixed-command calibration.
        # Keep the validated reference's estimator and transport convention.
        base.reference = {}
    env = TaskCurriculum(base,[dict(catch_goal=goal,seat_jitter_m=jitter)])
    actor = PPO.load(candidate,device='cpu') if candidate else None
    try:
        obs,_ = env.reset(seed=seed)
        while True:
            action = opening_action(obs,profile,probe)
            if actor is not None:
                action += reception_residual(obs,actor.predict(obs,deterministic=True)[0],cap)
            obs,_,done,truncated,info = env.step(action)
            if done or truncated:
                break
        print('[Opening trial]',seed,profile,probe,'goal',goal,'phase_policy',bool(candidate),info['reason'],flush=True)
        return dict(profile=profile_dict,seed=seed,jitter_m=jitter,probe_m_s=probe,goal=goal,
                    phase_policy=bool(candidate),result=info,
                    reception_plans=getattr(env.unwrapped.controller,'reception_plans',[]),
                    uncaught_braking_steps=getattr(env.unwrapped.controller,'uncaught_braking_steps',0),
                    uncaught_braking_first=getattr(env.unwrapped.controller,'uncaught_braking_first',None),
                    clearance=env.unwrapped.controller.clearance.copy())
    finally:
        env.close()


def metrics(rows):
    return dict(episodes=len(rows),completed=sum(bool(r['result'].get('training_prefix') or r['result']['is_success']) for r in rows),
                unsafe=sum(r['result']['reason'] in TaskResidualEnv.UNSAFE for r in rows),
                worst_air_gap_m=min((r['clearance']['min_gap_m'] for r in rows
                                    if r['clearance']['min_gap_m'] is not None),default=None),
                worst_hand_stand_gap_m=min((r['clearance']['min_hand_stand_gap_m'] for r in rows
                    if r['clearance'].get('min_hand_stand_gap_m') is not None),default=None),
                max_robot_penetration_m=max((r['clearance'].get('max_robot_penetration_m',0.) for r in rows),default=0.))


def rank(rows):
    m = metrics(rows)
    margin = min((x for x in (m['worst_air_gap_m'],m['worst_hand_stand_gap_m']) if x is not None),default=-1e9)
    return (-m['unsafe'],m['completed'],margin)


def summarize_results(profiles,chosen,test,phase,complete):
    tested = [0]+([chosen] if chosen else [])
    blocks = {str(i):metrics([r for r in test if r['profile']==profiles[i]]) for i in tested}
    center = [r for r in test if r['profile']==profiles[0] and np.all(np.asarray(r['probe_m_s'])==0.)]
    baseline,selected = blocks['0'],blocks[str(chosen)]
    clean_preserved = all(any(r['profile']==profiles[i] and r['result']['is_success'] for r in complete) for i in tested)
    accepted = bool(chosen and clean_preserved
                    and selected['completed']>baseline['completed'] and selected['unsafe']<=baseline['unsafe'])
    return dict(selected_profile=profiles[chosen],test=blocks,phase_baseline=metrics(center),
        phase_policy=metrics(phase),calibration_accepted=accepted,
        complete_clean=[dict(profile=r['profile'],success=r['result']['is_success'],reason=r['result']['reason']) for r in complete])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--train-seeds',type=int,nargs='+',default=[411100,411101])
    parser.add_argument('--validation-seeds',type=int,nargs='+',default=[411120,411121])
    parser.add_argument('--test-seeds',type=int,nargs='+',default=[411140,411141,411142,411143])
    parser.add_argument('--jitter-mm',type=float,default=.5)
    parser.add_argument('--probe-mm-s',type=float,default=1.)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--phase-policy-dir',type=Path)
    parser.add_argument('--phase-checkpoint',default='partial_epoch6000.zip')
    parser.add_argument('--parent-dir',type=Path)
    parser.add_argument('--reception-cap-mm',type=float,default=2.)
    search = parser.add_mutually_exclusive_group()
    search.add_argument('--local-velocity-grid',action='store_true')
    search.add_argument('--joint-opening',action='store_true')
    search.add_argument('--retained-height-grid',action='store_true')
    parser.add_argument('--reception-planner',choices=('off','audit','project'),default='off')
    parser.add_argument('--third-launch-dvz',type=float)
    parser.add_argument('--uncaught-braking',action='store_true')
    args = parser.parse_args()
    from .audit_task_learning import check_splits
    check_splits([args.train_seeds,args.validation_seeds,args.test_seeds])
    if (args.output_dir.exists() or args.jobs<1 or not 0.<args.jitter_mm<=10.
            or not 0.<args.probe_mm_s<=10. or not 0.<=args.reception_cap_mm<=60.):
        parser.error('Use new outputs and bounded positive experimental settings')
    profiles = [asdict(OpeningProfile(t,v)) for t,v in product((0.,.1,.2),(0.,.1))]
    if args.local_velocity_grid:
        profiles = [asdict(OpeningProfile())]+[asdict(OpeningProfile(velocity_bias_m_s=(x,0.,z)))
            for x,z in product((-.01,0.,.01),repeat=2) if x!=0. or z!=0.]
    if args.joint_opening:
        profiles = joint_profiles()
    if args.retained_height_grid:
        profiles = retained_height_profiles()
    if args.uncaught_braking and args.reception_planner!='off':
        parser.error('Evaluate uncaught braking separately from kinematic projection')
    if args.reception_planner!='off' or args.uncaught_braking:
        if args.local_velocity_grid or args.joint_opening or args.retained_height_grid:
            parser.error('Evaluate reception planning separately from a parameter search')
        profiles = [asdict(OpeningProfile()),asdict(OpeningProfile(reception_planner=args.reception_planner,
                                                                 uncaught_braking=args.uncaught_braking))]
        if args.third_launch_dvz is not None:
            velocity = args.third_launch_dvz
            launches = ((0.,0.,0.),(0.,0.,0.),(0.,0.,velocity))
            profiles += [asdict(OpeningProfile(launch_velocity_m_s=launches)),
                         asdict(OpeningProfile(launch_velocity_m_s=launches,reception_planner=args.reception_planner,
                                               uncaught_braking=args.uncaught_braking))]
    elif args.third_launch_dvz is not None:
        parser.error('--third-launch-dvz requires a reception planner or braking comparison')
    coordinated = args.joint_opening or args.retained_height_grid or args.reception_planner!='off' or args.uncaught_braking
    evaluation_goal = 6 if coordinated else 3
    probes = [[0.,0.,0.]]
    for axis,sign in product(range(3),(-1.,1.)):
        value = [0.,0.,0.]; value[axis] = sign*args.probe_mm_s/1000.; probes.append(value)
    inputs = {args.reference.as_posix():digest(args.reference),Path(__file__).as_posix():digest(Path(__file__))}
    candidate = None
    if args.phase_policy_dir:
        if args.parent_dir is None:
            parser.error('A phase policy requires its frozen parent manifest')
        _,parent = load_parent(args.parent_dir,'ppo_bc_seed0_epoch2000.zip',args.reference)
        _,hashes,reserved = load_corrections([args.phase_policy_dir],args.reference,args.parent_dir,
                                           'ppo_bc_seed0_epoch2000.zip',parent['goal'],parent['control_releases'])
        reserved |= set(parent['demonstration_attempt_seeds']+parent['validation_episode_seeds']+parent['test_episode_seeds'])
        if set(args.train_seeds+args.validation_seeds+args.test_seeds)&reserved:
            parser.error('Calibration initial states overlap earlier policy splits')
        inputs.update(hashes)
        selection_path = args.phase_policy_dir/'selection.json'
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        candidate = args.phase_policy_dir/args.phase_checkpoint
        matches = [r for r in selection['candidates'] if r['checkpoint']==args.phase_checkpoint]
        if len(matches)!=1 or digest(candidate)!=matches[0]['checkpoint_sha256']:
            parser.error('Require a frozen recorded policy checkpoint')
        inputs.update({candidate.as_posix():digest(candidate),selection_path.as_posix():digest(selection_path)})
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'driver_source.py').write_bytes(Path(__file__).read_bytes())
    protocol = dict(scope='Opening calibration and restricted legacy-policy pilot; not trained RL or full robustness acceptance',
        source_sha256=runtime_sources(),input_sha256=inputs,profiles=profiles,probes_m_s=probes,
        train_seeds=args.train_seeds,validation_seeds=args.validation_seeds,test_seeds=args.test_seeds,
        jitter_m=args.jitter_mm/1000.,goal=evaluation_goal,training_goal=3,reception_cap_m=args.reception_cap_mm/1000.,
        monitoring='Confirmed airborne balls without either hand contact; read-only world-state clearance',
        phase_policy_scope='Original zero-settle baseline only; not a policy trained with this restriction',
        local_velocity_grid=args.local_velocity_grid,
        joint_opening=args.joint_opening,
        retained_height_grid=args.retained_height_grid,
        uncaught_braking=args.uncaught_braking,
        observation_reference='Estimator from validated reference; innovation gate disabled only for fixed calibration/probes',
        selection='Preserve clean center, shortlist by training safety/completion, validate command probes, freeze before test')
    write_new(args.output_dir/'protocol.json',protocol)
    def job(profile,seed,jitter,probe,goal=evaluation_goal,actor=None):
        return (args.reference,profile,seed,jitter,probe,goal,actor,args.reception_cap_mm/1000.)
    scenarios = [(0,0.)]+[(s,args.jitter_mm/1000.) for s in args.train_seeds]
    train = run_jobs(trial,[job(p,s,j,[0.]*3,goal=3) for p in profiles for s,j in scenarios],args.jobs,args.output_dir,'train')
    grouped = [[r for r in train if r['profile']==p] for p in profiles]
    eligible = [i for i,rows in enumerate(grouped) if rows[0]['result'].get('training_prefix')]
    if 0 not in eligible:
        raise ValueError('The unchanged clean baseline must reproduce its prefix')
    shortlist = [0]+[i for i in sorted(eligible,key=lambda i:rank(grouped[i]),reverse=True) if i!=0][:2]
    write_new(args.output_dir/'shortlist.json',dict(indices=shortlist,training_metrics=[metrics(rows) for rows in grouped]))
    validation_jobs = [job(profiles[i],s,args.jitter_mm/1000.,probe)
                       for i in shortlist for s in args.validation_seeds for probe in probes]
    if coordinated:
        validation_jobs += [job(profiles[i],0,0.,[0.]*3) for i in shortlist]
    validation = run_jobs(trial,validation_jobs,
                          args.jobs,args.output_dir,'validation')
    eligible_validation = [i for i in shortlist if not coordinated or any(
        r['profile']==profiles[i] and r['jitter_m']==0. and r['result'].get('training_prefix') for r in validation)]
    if 0 not in eligible_validation:
        raise ValueError('Unchanged baseline must reproduce the clean validation goal')
    chosen = max(eligible_validation,key=lambda i:rank([r for r in validation if r['profile']==profiles[i] and r['jitter_m']>0.]))
    write_new(args.output_dir/'selection.json',dict(index=chosen,profile=profiles[chosen],
        validation_metrics={str(i):metrics([r for r in validation if r['profile']==profiles[i]]) for i in shortlist}))
    test_profiles = [0]+([chosen] if chosen else [])
    if args.uncaught_braking:
        # A safety intervention must also be checked on missed-catch cases
        # excluded by successful-prefix selection. Freeze all pairs before test.
        test_profiles = list(range(len(profiles)))
    test = run_jobs(trial,[job(profiles[i],s,args.jitter_mm/1000.,probe)
                          for i in test_profiles for s in args.test_seeds for probe in probes],args.jobs,args.output_dir,'test')
    # The learned policy keeps its original observation/reference convention.
    # Its restriction is evaluated separately from any shifted opening schedule.
    phase = run_jobs(trial,[job(profiles[0],s,args.jitter_mm/1000.,[0.]*3,actor=candidate)
                           for s in args.test_seeds],args.jobs,args.output_dir,'phase') if candidate else []
    complete = run_jobs(trial,[job(profiles[i],0,0.,[0.]*3,goal=30) for i in test_profiles],
                        args.jobs,args.output_dir,'complete_clean')
    for name,expected in inputs.items():
        if digest(Path(name))!=expected:
            raise ValueError('Calibration input changed: '+name)
    if runtime_sources()!=protocol['source_sha256']:
        raise ValueError('Calibration runtime changed')
    summary = summarize_results(profiles,chosen,test,phase,complete)
    if args.uncaught_braking:
        summary['braking_pairs'] = [dict(
            baseline_profile=i,braking_profile=i+1,
            baseline=metrics([r for r in test if r['profile']==profiles[i]]),
            braking=metrics([r for r in test if r['profile']==profiles[i+1]]))
            for i in range(0,len(profiles),2)]
    write_new(args.output_dir/'report.json',dict(protocol=protocol,training=train,validation=validation,test=test,
        phase_policy=phase,complete_clean=complete,summary=summary,rl_gain_verified=False))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
