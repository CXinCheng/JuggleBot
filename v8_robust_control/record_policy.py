"""Record a selected PPO checkpoint on a physical single- or multi-ball task."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import torch
from stable_baselines3 import PPO

from .compare_residual import digest, read_json
from .compare_videos import filter_path
from .transfer_env import TransferEnv
from .cascade_env import CascadeEnv
from . import train_transfer, train_cascade
from v6_humanoid_hands.juggle import cascade_camera


def checked_checkpoint(folder, selected):
    folder = Path(folder).resolve()
    checkpoint = (folder/selected['checkpoint']).resolve()
    if not checkpoint.is_relative_to(folder):
        raise ValueError('Checkpoint must stay inside the model directory')
    if not checkpoint.is_file():
        raise ValueError(f'Selected checkpoint not found: {checkpoint}. Regenerate the corresponding training run.')
    if digest(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the selected training record')
    return checkpoint


def load_selected(training, folder, seed):
    manifest = training['manifest']
    single = manifest['scope'] == 'complete_single_ball_residual_validation'
    if not single and manifest['scope'] != 'cascade_residual_validation':
        raise ValueError('Require a complete single-ball or cascade training archive')
    driver = train_transfer if single else train_cascade
    current = driver.sources()
    compatible = current == manifest['source_sha256'] if single else train_cascade.runtime_compatible(
        manifest['source_sha256'], current)
    if not compatible or driver.summarize(training) != training['summary']:
        raise ValueError('Training archive or runtime differs')
    selected = next(row for row in training['models'] if row['seed'] == seed)
    checkpoint = checked_checkpoint(folder, selected)
    actor = PPO.load(checkpoint, device='cpu')
    if actor.num_timesteps != selected['selected_step']:
        raise ValueError('Policy training budget differs')
    if not single and (actor.juggling_source_sha256 != manifest['source_sha256']
            or actor.juggling_task != manifest['task']
            or actor.juggling_throws != manifest['requested_throws']
            or actor.juggling_ball_count != manifest['ball_count']
            or actor.juggling_action_layout != manifest['action_layout']
            or actor.juggling_observation_layout != manifest['observation_layout']):
        raise ValueError('Policy task or interface differs')
    reference = Path(manifest['reference_dir'])
    if digest(reference/'manifest.json') != manifest['reference_manifest_sha256']:
        raise ValueError('Reference manifest differs from training')
    if not single and digest(reference/'reference.npz') != manifest['reference_sha256']:
        raise ValueError('Reference trajectory differs from training')
    return actor, selected, current, single


def run_episode(env, actor, seed, capture):
    """Capture reset and every executed control step, including the final state."""
    observation, _ = env.reset(seed=seed)
    capture()
    frames = 1
    while True:
        action, _ = actor.predict(observation, deterministic=True)
        observation, _, done, truncated, info = env.step(action)
        capture()
        frames += 1
        if done or truncated:
            return info, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--training-seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--case', choices=list(train_transfer.CASES), default='clean')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Episode and video metadata JSON')
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    if (args.seed < 0 or args.video.suffix.lower() != '.mp4'
            or args.output.suffix.lower() != '.json'
            or args.video.exists() or args.output.exists()
            or args.video.resolve() == args.output.resolve()):
        parser.error('Use a nonnegative seed and distinct new MP4 and JSON paths')
    if not args.training.is_file():
        parser.error(f'Training archive not found: {args.training}. '
                     'Run train_transfer run (single ball) or train_cascade run (two/three balls) '
                     'to generate the training archive and checkpoints before recording.')
    if not args.input_dir.is_dir():
        parser.error(f'Checkpoint directory not found: {args.input_dir}. '
                     'Generate the corresponding training run before recording.')
    fonts = [args.font] if args.font else [Path('C:/Windows/Fonts/arial.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
        Path('/System/Library/Fonts/Supplemental/Arial.ttf')]
    font = next((path for path in fonts if path and path.is_file()), None)
    if font is None:
        parser.error('Supply an installed TrueType font with --font')
    torch.set_num_threads(1)
    try:
        training_hash = digest(args.training)
        training = read_json(args.training)
        actor, selected, sources, single = load_selected(training, args.input_dir, args.training_seed)
        checkpoint = checked_checkpoint(args.input_dir, selected)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    manifest = training['manifest']
    reference = Path(manifest['reference_dir'])
    env = TransferEnv(reference) if single else CascadeEnv(reference)
    env.config = replace(env.config, **train_transfer.CASES[args.case])
    task = 'single' if single else env.task
    fps = round(1/env.config.control_period_s)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    renderer = None
    try:
        renderer = mujoco.Renderer(env.model, height=720, width=1280)
        camera = cascade_camera(env.controller.torso_stand_height)
        with tempfile.TemporaryDirectory(prefix='v8_policy_video_') as temporary:
            raw = Path(temporary)/'episode.mp4'
            with imageio.get_writer(raw, fps=fps, codec='libx264', quality=8) as writer:
                def capture():
                    renderer.update_scene(env.data, camera=camera)
                    writer.append_data(renderer.render())
                result, frames = run_episode(env, actor, args.seed, capture)
            flights = 1 if single and result['is_success'] else result.get('catches', 0)
            requested = 1 if single else env.target_throws
            caption = Path(temporary)/'caption.txt'
            caption.write_text(
                f'Nominal controller + PPO residual | {task} | {args.case} | episode seed {args.seed}\n'
                f'Training seed {args.training_seed} | selected step {selected["selected_step"]} | '
                f'flights {flights}/{requested}\n'
                f'Stop: {result["reason"]} at {result["simulated_time_s"]:.3f} s | normal speed',
                encoding='utf-8', newline='\n')
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                '-nostdin', '-n', '-i', str(raw), '-vf',
                "drawbox=x=0:y=0:w=iw:h=82:color=black@0.65:t=fill,"
                f"drawtext=fontfile='{filter_path(font)}':textfile='{filter_path(caption)}':"
                'expansion=none:fontcolor=white:fontsize=20:x=12:y=8:line_spacing=5',
                '-an', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', str(args.video)], check=True)
        driver = train_transfer if single else train_cascade
        if (driver.sources() != sources or digest(args.training) != training_hash
                or digest(checkpoint) != selected['checkpoint_sha256']):
            raise ValueError('Policy or sources changed while recording')
        metadata = dict(schema_version=1, task=task, case=args.case, episode_seed=args.seed,
            controller='nominal controller + trained PPO residual', deterministic=True,
            training=args.training.as_posix(), training_sha256=training_hash,
            training_seed=args.training_seed, selected_step=selected['selected_step'],
            checkpoint=checkpoint.relative_to(args.input_dir.resolve()).as_posix(),
            checkpoint_sha256=selected['checkpoint_sha256'], source_sha256=sources,
            recorder_sha256=digest(Path(__file__)),
            reference_sha256=digest(reference/'reference.npz'),
            reference_manifest_sha256=digest(reference/'manifest.json'),
            video=dict(path=args.video.as_posix(), sha256=digest(args.video), frames=frames,
                       fps=fps, speed=1., duration_s=frames/fps, end_hold_s=0.,
                       width=1280, height=720), result=result,
            interpretation='A demonstration episode, not an independent robustness estimate.')
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
        print(f'{task}: {result["reason"]}; {frames} frames; {args.video}', flush=True)
        if not result['is_success']:
            raise SystemExit(1)
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == '__main__':
    main()
