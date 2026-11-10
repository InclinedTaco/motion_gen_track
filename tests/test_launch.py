import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('motion', ROOT / 'scripts/motion.py')
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)


class LaunchTests(unittest.TestCase):
    def args(self, command='track', extra=()):
        return motion.parser().parse_args([command, '--checkpoint', 'weights/model.pt', '--stats', 'weights/stats.json', *extra])

    def test_live_tracking_is_inference_and_explicit_checkpoint(self):
        args = self.args(extra=['--composer', 'weights with spaces/composer.pth'])
        cwd, cmd = motion.build_command(args)
        self.assertEqual(cwd, ROOT / 'tracking')
        self.assertIn('test=True', cmd)
        self.assertIn('no_log=True', cmd)
        self.assertIn('epoch=0', cmd)
        load = next(x.split('=', 1)[1] for x in cmd if x.startswith('learning.params.load_path='))
        self.assertEqual(json.loads(load), args.composer)
        self.assertIn('learning.params.load_checkpoint=True', cmd)
        self.assertNotIn('--config-name', cmd)

    def test_paths_resolved_before_child_changes_cwd(self):
        args = self.args('generate')
        cwd, cmd = motion.build_command(args)
        self.assertTrue(Path(args.checkpoint).is_absolute())
        self.assertTrue((cwd / cmd[1]).is_file())
        self.assertEqual(cmd[cmd.index('--checkpoint') + 1], args.checkpoint)

    def test_hydra_paths_escape_special_characters(self):
        path = 'models/a,b [trial]/"composer".pth'
        self.assertEqual(json.loads(motion.hydra_string(path)), path)

    def test_dry_run_needs_no_model_or_simulator(self):
        self.assertEqual(motion.main(['track', '--checkpoint', '/absent/model.pt', '--stats', '/absent/stats.json', '--dry-run']), 0)

    def test_asset_validation_rejects_wrong_schema_and_stride(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / 'best.pt'; ckpt.touch()
            stats = Path(tmp) / 'stats.json'
            args = motion.parser().parse_args(['generate','--checkpoint',str(ckpt),'--stats',str(stats)])
            stats.write_text(json.dumps({'horizon':15,'target_mean':[0]*1140,'condition_schema':'phase'}))
            motion.validate_assets(args)
            args.stride = 16
            with self.assertRaisesRegex(ValueError, 'stride'): motion.validate_assets(args)
            args.stride = 5
            stats.write_text(json.dumps({'horizon':15,'target_mean':[0]*1275,'condition_schema':'object_place'}))
            with self.assertRaisesRegex(ValueError, '76D'): motion.validate_assets(args)

    def test_missing_assets_fail_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'Missing assets'):
            motion.validate_assets(self.args())

    def test_source_layout_and_case_safety(self):
        paths = [str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
        self.assertEqual(len(paths), len(set(p.casefold() for p in paths)))
        for name in ['generation/flow_matching/__init__.py','generation/examples/joint_flow/model.py',
                     'tracking/phc/run_hydra.py','tracking/scripts/flow_server_x15.py']:
            self.assertTrue((ROOT / name).is_file())
        for p in (ROOT / 'tracking/scripts').glob('flow*.py'):
            source = p.read_text()
            self.assertNotIn(' / "flow_matching"', source, p.name)


if __name__ == '__main__':
    unittest.main()
