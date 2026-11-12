"""Compose labelled, normal-speed perception clips from evaluated trial records."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg

from .summarize_perception import validate


def filter_path(path):
    return str(Path(path).resolve()).replace('\\','/').replace(':','\\:').replace("'", "'\\''")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',default=['baseline','noise_3mm','combined'])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata-output',type=Path,required=True,help='New JSON path for comparison provenance.')
    parser.add_argument('--hold-seconds',type=float,default=0.,help='Optional final-frame hold after the longest clip (seconds; default 0).')
    parser.add_argument('--font',type=Path,help='TrueType font file for captions.')
    args = parser.parse_args()
    if not 1 <= len(args.cases) <= 3 or len(set(args.cases)) != len(args.cases):
        parser.error('Select one to three distinct cases')
    if not math.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        parser.error('--hold-seconds must be finite and nonnegative')
    if len({p.resolve() for p in (args.input,args.output,args.metadata_output)}) != 3:
        parser.error('Input, video and metadata paths must be distinct')
    if args.output.exists() or args.metadata_output.exists():
        parser.error('Comparison video and metadata paths must be new')
    try:
        _,trials = validate([json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()])
    except (ValueError,KeyError,TypeError) as error:
        parser.error(str(error))
    chosen = []
    for name in args.cases:
        row = next((r for r in trials if r['case']==name and r['seed']==args.seed),None)
        if row is None or 'video' not in row or not Path(row['video']['path']).is_file():
            parser.error(f'No recorded video for {name}, seed {args.seed}')
        chosen.append(row)
    fonts = [args.font] if args.font else [Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),Path('/System/Library/Fonts/Supplemental/Arial.ttf')]
    font = next((p for p in fonts if p and p.is_file()),None)
    if font is None:
        parser.error('Supply an installed TrueType font with --font')
    duration = max(r['video']['frames']/r['video']['fps'] for r in chosen)+args.hold_seconds
    command = [imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-n']
    for row in chosen:
        command += ['-i',row['video']['path']]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v8_captions_') as temp:
        filters = []
        for index,row in enumerate(chosen):
            c,r = row['perception'],row['result']
            reason = 'completed' if r['success'] else (r.get('failure_event') or {}).get('reason','failed')
            label = (f"Classical controller | {row['case']} | seed {row['seed']}\n"
                     f"sigma {c['noise_std_m']*1000:g} mm | delay {c['delay_s']*1000:g} ms | dropout {c['dropout_probability']*100:g}%\n"
                     f"Result: {r['validated_flights']}/{r['requested_throws']} flights | drops {r['drops']} | ball collisions {r['ball_contacts']}\n"
                     f"Stop: {reason} at {r['time']:.3f} s")
            caption = Path(temp)/f'caption{index}.txt'
            caption.write_text(label,encoding='utf-8',newline='')
            raw_duration = row['video']['frames']/row['video']['fps']
            filters.append(f"[{index}:v]setpts=PTS-STARTPTS,scale=640:360,fps=60,tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},pad=640:480:0:0:black,drawtext=fontfile='{filter_path(font)}':textfile='{filter_path(caption)}':expansion=none:fontcolor=white:fontsize=16:x=12:y=372:line_spacing=5,drawtext=fontfile='{filter_path(font)}':text='STOPPED - final frame held':fontcolor=yellow:fontsize=18:x=12:y=15:enable='gte(t,{raw_duration})'[v{index}]")
        if len(chosen)>1:
            filters.append(''.join(f'[v{i}]' for i in range(len(chosen)))+f'hstack=inputs={len(chosen)}[out]')
            output_label = '[out]'
        else:
            output_label = '[v0]'
        command += ['-filter_complex',';'.join(filters),'-map',output_label,'-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-r','60',str(args.output)]
        subprocess.run(command,check=True)
    metadata = dict(input=args.input.as_posix(),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    output=args.output.as_posix(),cases=args.cases,seed=args.seed,speed=1.,fps=60,duration_s=duration,
                    end_hold_s=args.hold_seconds,
                    end_behavior='Final frames held and labelled after each episode stops; no simulated continuation.',
                    trials=[dict(case=row['case'],seed=row['seed'],video=row['video'],
                                 result={key:row['result'][key] for key in ('success','validated_flights','requested_throws','time','drops','ball_contacts','error')})
                            for row in chosen])
    args.metadata_output.parent.mkdir(parents=True,exist_ok=True)
    with args.metadata_output.open('x',encoding='utf-8',newline='') as stream:
        stream.write(json.dumps(metadata,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
