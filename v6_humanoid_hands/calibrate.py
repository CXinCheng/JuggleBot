"""Validate and export LEAP optimizer inputs with a completed physical run.

No file is written unless every requested flight has been caught. Ordinary
juggling runs never update their input seeds. Run from the repository root.
"""
import argparse
import json
from pathlib import Path

from jugglebot.humanoid.cascade import ContactCascade, DEFAULT_THROW_INTERVAL_S, DEFAULT_FLIGHT_TIME_S


class CalibrationCascade(ContactCascade):
    def log(self, kind, side, ball, **extra):
        super().log(kind, side, ball, **extra)
        if not self.silent and kind == 'release' and extra['count'] % 10 == 0:
            style = self.style_summary()
            print(f"[Progress] {extra['count']} releases | max height {self.max_height:.3f} m | "
                  f"min spacing {self.minimum_distance:.3f} m | "
                  f"wrist pitch max {style['wrist_pitch_max_deg']:.1f} deg", flush=True)

    def candidate_inputs(self):
        """Cache direct successes, retaining the original guesses for retries.

        A retry's final command belongs to a later physical state. Reusing it
        at the first attempt can change the cascade even with fixed forecasts.
        The resulting candidate still requires an independent physical run.
        """
        execution = dict(self.execution_warm)
        attempts = {}
        count = 0
        for event in self.events:
            if event['kind'] == 'release':
                count = event['count']
            elif event['kind'] in ('plan', 'plan_rejected'):
                key = f"throw:{event['side']}:{count}"
                attempts.setdefault(key, []).append(event)
            elif event['kind'] == 'return_plan':
                score = event.get('score', float(any(event['error'])))
                if score == 0.:
                    execution[f"return:{event['side']}:{count}"] = event['parameters']
        for key, plans in attempts.items():
            if len(plans) == 1 and plans[0].get('score', float('inf')) < .02:
                execution[key] = plans[0]['command']
        return dict(forecast=dict(self.warm), execution=execution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--balls', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--throws', type=int, default=50)
    parser.add_argument('--timeout', type=float, default=35.)
    parser.add_argument('--tau', type=float, default=DEFAULT_THROW_INTERVAL_S, help='Alternating throw interval (s).')
    parser.add_argument('--flight', type=float, default=DEFAULT_FLIGHT_TIME_S, help='Separation-to-palm-interception target time (s).')
    parser.add_argument('--output', type=Path, required=True, help='Explicit destination for accepted optimizer inputs.')
    parser.add_argument('--initial-plans', type=Path, help='Optional optimizer initial guesses for this calibration run.')
    args = parser.parse_args()
    if args.throws <= 0 or args.timeout <= 0 or args.tau <= 0 or args.flight <= 0:
        parser.error('throws, timeout, tau and flight must be positive')
    timing = dict(tau=args.tau, flight=args.flight)
    controller = CalibrationCascade(n_balls=args.balls, verbose=True, **timing)
    if args.initial_plans is not None:
        controller.set_optimizer_inputs(json.loads(args.initial_plans.read_text(encoding='utf-8')))
    initial_plans = dict(forecast=dict(controller.warm), execution=dict(controller.execution_warm))
    result = controller.run_continuous(throws=args.throws, timeout=args.timeout, guard=True)
    print(json.dumps(result, indent=2), flush=True)
    if not result['success']:
        raise SystemExit('Calibration failed; optimizer input file was not written.')
    candidate = controller.candidate_inputs()
    accepted = initial_plans
    if candidate != initial_plans:
        # Forecasts remain fixed. Optimized live guesses still need an
        # independent physical run before they can be exported as accepted.
        print('Independently validating optimized live guesses...', flush=True)
        verification = CalibrationCascade(n_balls=args.balls, verbose=True, **timing)
        verification.set_optimizer_inputs(candidate)
        checked = verification.run_continuous(throws=args.throws, timeout=args.timeout, guard=True)
        print(json.dumps(checked, indent=2), flush=True)
        if checked['success']:
            accepted = candidate
        else:
            print('Optimized guesses failed verification; retaining the validated original inputs.', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted = dict(accepted, conditions=controller.conditions_summary())
    args.output.write_text(json.dumps(accepted, indent=2)+'\n', encoding='utf-8')
    print(f"Saved {len(accepted['forecast'])} forecast and {len(accepted['execution'])} live optimizer inputs to {args.output}")


if __name__ == '__main__':
    main()
