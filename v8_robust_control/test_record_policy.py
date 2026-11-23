"""Recording preserves selected weights, policy actions and terminal frames."""
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest.mock import patch

from .compare_residual import digest
from .record_policy import checked_checkpoint, run_episode, main


class PolicyRecordingTests(unittest.TestCase):
    def test_checkpoint_must_match_selected_weights_and_stay_inside_model_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/'models'
            folder.mkdir()
            path = folder/'ppo.zip'
            path.write_bytes(b'selected policy')
            row = dict(checkpoint=path.name, checkpoint_sha256=digest(path))
            self.assertEqual(checked_checkpoint(folder, row), path.resolve())
            path.write_bytes(b'different policy')
            with self.assertRaises(ValueError):
                checked_checkpoint(folder, row)
            outside = root/'outside.zip'
            outside.write_bytes(b'outside')
            with self.assertRaises(ValueError):
                checked_checkpoint(folder, dict(checkpoint='../outside.zip', checkpoint_sha256=digest(outside)))
            with self.assertRaisesRegex(ValueError, 'Selected checkpoint not found'):
                checked_checkpoint(folder, dict(checkpoint='missing.zip', checkpoint_sha256='missing'))

    def test_records_actual_policy_actions_and_final_state_for_success_failure_or_truncation(self):
        for reason, truncated in [('SUCCESS', False), ('FORBIDDEN_CONTACT', False), ('LIMIT', True)]:
            with self.subTest(reason=reason):
                class Env:
                    def reset(self, seed):
                        self.seed, self.time, self.actions = seed, 0, []
                        return 0, {}

                    def step(self, action):
                        self.actions.append(action)
                        self.time += 1
                        return self.time, 0., self.time == 2 and not truncated, self.time == 2 and truncated, dict(reason=reason)

                class Actor:
                    def predict(self, observation, deterministic):
                        self.deterministic = deterministic
                        return observation+.25, None

                env, actor, captured = Env(), Actor(), []
                result, frames = run_episode(env, actor, 17, lambda: captured.append(env.time))
                self.assertEqual(env.seed, 17)
                self.assertEqual(env.actions, [.25, 1.25])
                self.assertEqual(captured, [0, 1, 2])
                self.assertEqual(frames, 3)
                self.assertEqual(result['reason'], reason)
                self.assertTrue(actor.deterministic)

    def test_existing_video_or_metadata_is_rejected_before_loading_a_policy(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for existing in ('clip.mp4', 'clip.json'):
                with self.subTest(existing=existing):
                    path = folder/existing
                    path.write_bytes(b'preserve')
                    args = ['record_policy', '--training', 'missing.json', '--input-dir', 'missing',
                            '--video', str(folder/'clip.mp4'), '--output', str(folder/'clip.json')]
                    with patch('sys.argv', args), redirect_stderr(StringIO()), patch('v8_robust_control.record_policy.load_selected') as load:
                        with self.assertRaises(SystemExit) as failure:
                            main()
                    self.assertEqual(failure.exception.code, 2)
                    load.assert_not_called()
                    self.assertEqual(path.read_bytes(), b'preserve')
                    path.unlink()

    def test_missing_training_inputs_explain_the_prerequisite_without_creating_outputs(self):
        for missing in ('training', 'checkpoints'):
            with self.subTest(missing=missing), TemporaryDirectory() as temporary:
                folder = Path(temporary)
                training, models = folder/'training.json', folder/'models'
                if missing == 'training':
                    models.mkdir()
                else:
                    training.write_text('{}', encoding='utf-8')
                video, metadata = folder/'clip.mp4', folder/'clip.json'
                args = ['record_policy', '--training', str(training), '--input-dir', str(models),
                        '--video', str(video), '--output', str(metadata)]
                error_text = StringIO()
                with patch('sys.argv', args), redirect_stderr(error_text), patch('v8_robust_control.record_policy.load_selected') as load:
                    with self.assertRaises(SystemExit) as failure:
                        main()
                self.assertEqual(failure.exception.code, 2)
                expected = 'Training archive not found' if missing == 'training' else 'Checkpoint directory not found'
                self.assertIn(expected, error_text.getvalue())
                self.assertIn('before recording', error_text.getvalue())
                load.assert_not_called()
                self.assertFalse(video.exists())
                self.assertFalse(metadata.exists())


if __name__ == '__main__':
    unittest.main()
