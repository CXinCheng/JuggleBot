"""Video presentation duration and metadata destinations, without simulation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from .compare_videos import main


class TestVideoPresentation(unittest.TestCase):
    def test_short_clip_retains_episode_result_and_uses_separate_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'trials.jsonl'
            source.write_text('{}\n',encoding='utf-8')
            raw=root/'raw.mp4'
            raw.write_bytes(b'raw video fixture')
            output=root/'videos/presentation.mp4'
            metadata=root/'results/video.json'
            result=dict(success=False,validated_flights=0,requested_throws=30,time=.279,
                        drops=0,ball_contacts=0,error='NO_FEASIBLE_RECEPTION',
                        failure_event=dict(reason='NO_FEASIBLE_RECEPTION'))
            trial=dict(case='noise_3mm',seed=0,result=result,
                       perception=dict(noise_std_m=.003,delay_s=0.,dropout_probability=0.),
                       video=dict(path=str(raw),frames=17,fps=60,speed=1.))
            argv=['compare','--input',str(source),'--cases','noise_3mm',
                  '--output',str(output),'--metadata-output',str(metadata),'--font',str(Path(__file__))]
            with patch.object(sys,'argv',argv), patch('v8_robust_control.compare_videos.validate',return_value=({},[trial])), \
                    patch('v8_robust_control.compare_videos.imageio_ffmpeg.get_ffmpeg_exe',return_value='ffmpeg'), \
                    patch('v8_robust_control.compare_videos.subprocess.run') as run:
                main()
            record=json.loads(metadata.read_text(encoding='utf-8'))
            self.assertAlmostEqual(record['duration_s'],17/60)
            self.assertEqual(record['end_hold_s'],0.)
            self.assertEqual(record['trials'][0]['result']['time'],.279)
            self.assertEqual(record['trials'][0]['video']['frames'],17)
            self.assertEqual(raw.read_bytes(),b'raw video fixture')
            self.assertFalse(output.with_suffix('.json').exists())
            self.assertIn('STOPPED - final frame held',run.call_args.args[0][run.call_args.args[0].index('-filter_complex')+1])

    def test_invalid_hold_and_path_aliases_rejected_before_rendering(self):
        for extra in (['--hold-seconds','-1'],['--hold-seconds','nan'],
                      ['--metadata-output','input.jsonl'],['--metadata-output','movie.mp4']):
            with self.subTest(extra=extra), patch.object(sys,'argv',['compare','--input','input.jsonl',
                 '--output','movie.mp4','--metadata-output','metadata.json',*extra]), \
                    patch('v8_robust_control.compare_videos.subprocess.run') as run, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code,2)
                run.assert_not_called()

    def test_existing_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            metadata=Path(folder)/'record.json'
            metadata.write_text('original',encoding='utf-8')
            with patch.object(sys,'argv',['compare','--input','input.jsonl','--output',str(Path(folder)/'new.mp4'),
                              '--metadata-output',str(metadata)]),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(metadata.read_text(encoding='utf-8'),'original')


if __name__=='__main__':
    unittest.main()
