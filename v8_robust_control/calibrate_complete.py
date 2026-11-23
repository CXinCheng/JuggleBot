"""Calibrate full-task optimizer guesses using verified contact simulations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .complete_task import CompleteCascade, DEFAULT_SEED_PLANS, complete_record


def optimizer_inputs(controller):
    return {group:{key:np.asarray(value,dtype=float).tolist() for key,value in values.items()}
            for group,values in (('forecast',controller.warm),('execution',controller.execution_warm))}


def collect(controller, inputs):
    """Cache executed launches and zero-error receptions; keep forecasts fixed."""
    execution=dict(inputs['execution'])
    count=0
    launches={}
    receptions={}
    for event in controller.events:
        side=event['side']
        if event['kind']=='release':
            count=event['count']
        elif event['kind']=='plan':
            launches[side]=event
        elif event['kind']=='stroke':
            plan=launches.get(side)
            if plan and plan['time']==event['time'] and plan['ball']==event['ball']:
                execution[f'throw:{side}:{count}']=list(plan['command'])
        elif event['kind']=='return_plan':
            receptions[side]=event
        elif event['kind']=='prepare':
            plan=receptions.get(side)
            if (plan and plan['time']==event['time'] and plan['ball']==event['ball']
                    and plan.get('score')==0. and plan.get('tilt_scale',1.)==1.):
                execution[f'return:{side}:{count}']=list(plan['parameters'])
    return dict(forecast=inputs['forecast'],execution=execution)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_SEED_PLANS)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--executed-output',type=Path,
                   help='Export executed commands after native acceptance; requires separate event replay')
    p.add_argument('--throws',type=int,default=30)
    p.add_argument('--timeout',type=float,default=40.)
    p.add_argument('--max-iterations',type=int,default=8)
    args=p.parse_args()
    if args.throws<4 or args.throws%2 or args.max_iterations<1 or not np.isfinite(args.timeout) or args.timeout<=0:
        p.error('Use even throws >= 4, positive iterations and a finite positive timeout')
    if args.output.resolve()==args.report.resolve() or (args.input and args.input.resolve() in (args.output.resolve(),args.report.resolve())):
        p.error('Input, output and report paths must be distinct')
    if args.output.exists() or args.report.exists():
        p.error('Output and report paths must be new')
    if args.executed_output is not None:
        if (args.executed_output.resolve() in {path.resolve() for path in (args.input,args.output,args.report)}
                or args.executed_output.exists()):
            p.error('Executed command output must be a new, distinct path')
    inputs=json.loads(args.input.read_text(encoding='utf-8'))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x',encoding='utf-8',newline='') as stream:
        for iteration in range(args.max_iterations):
            controller=CompleteCascade(throws=args.throws,optimizer_inputs=inputs)
            controller.progress_interval=5
            effective=optimizer_inputs(controller)
            record=complete_record(None,args.throws,args.timeout,optimizer_file=None)
            record['calibration_input']=effective
            record['calibration_input_sha256']=hashlib.sha256(json.dumps(effective,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            record['base_optimizer_inputs_sha256']=record['optimizer_inputs_sha256']
            record['optimizer_inputs_sha256']=record['calibration_input_sha256']
            record['optimizer_inputs_encoding']='canonical-json'
            record['iteration']=iteration
            result=controller.run_complete(args.timeout)
            record['result']=result
            updated=collect(controller,effective) if not result['success'] else None
            if updated is not None:
                record['next_calibration_input']=updated
            stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
            print(f"[Calibration {iteration}] {result['validated_flights']}/{args.throws}, success={result['success']}, error={result['error']}",flush=True)
            if result['success']:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                with args.output.open('x',encoding='utf-8') as output:
                    output.write(json.dumps(effective,indent=2,allow_nan=False)+'\n')
                if args.executed_output is not None:
                    args.executed_output.parent.mkdir(parents=True,exist_ok=True)
                    with args.executed_output.open('x',encoding='utf-8') as output:
                        output.write(json.dumps(collect(controller,effective),indent=2,allow_nan=False)+'\n')
                return
            if updated==effective:
                raise SystemExit('No new feasible optimizer guesses; inspect the recorded task failure')
            inputs=updated
    raise SystemExit('No full-task input set accepted within the iteration limit')


if __name__=='__main__':
    main()
