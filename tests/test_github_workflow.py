"""Offline tests; fake transport is used, no real GPU score is asserted."""
import ast
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_github_notebook import CONFIG, make_notebook
from github_store import GitHubStore, safe_target, sha256
from github_data import download_archive
from marso_github import GitHubExperiment, source_bundle, atomic_checkpoint_patch
from next_pick_sampling import stable_grasp_events, sampling_weights
from next_pick_diagnostics import progress_summary


class MemoryStore(GitHubStore):
    def __init__(self):
        super().__init__('owner/repo', 'run-test', 'secret-not-for-serialization')
        self.content = {}
        self.failure = False
    def load(self, create=True):
        self.release = {'id': 1}
        return True
    def upload(self, path, name):
        if self.failure and name.startswith('snapshot-'):
            raise RuntimeError('simulated dropped connection')
        self.content[name] = Path(path).read_bytes()
        self.assets[name] = dict(name=name, size=len(self.content[name]), url=name)
        return self.assets[name]
    def download(self, asset, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.content[asset['name']])


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        self.store = MemoryStore()

    def put(self, name, content=b'model-weights'):
        path = self.source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_complete_snapshots_restore_exact_contents_and_skip_temporary_files(self):
        self.put('easy/checkpoints/model.pt')
        self.put('easy/checkpoints/model.pt.tmp', b'unfinished')
        self.put('easy/metrics.json', b'{"complete": true}')
        config = self.put('config.json', b'{"run_name": "demo"}')
        self.store.sync(self.source, 'easy')
        self.store.sync(self.source, 'common', [config])
        destination = self.root/'restore'
        self.assertEqual(self.store.restore(destination), 3)
        self.assertEqual((destination/'easy/checkpoints/model.pt').read_bytes(), b'model-weights')
        self.assertFalse((destination/'easy/checkpoints/model.pt.tmp').exists())
        self.assertEqual(self.store.restore(destination), 0)
        all_bytes = b''.join(self.store.content.values())
        self.assertNotIn(b'secret-not-for-serialization', all_bytes)

    def test_interrupted_upload_keeps_previous_snapshot(self):
        path = self.put('hard/checkpoints/model.pt', b'previous')
        self.store.sync(self.source, 'hard')
        path.write_bytes(b'new-model')
        self.store.failure = True
        with self.assertRaises(RuntimeError):
            self.store.sync(self.source, 'hard')
        restored = self.root/'restore'
        self.store.restore(restored)
        self.assertEqual((restored/'hard/checkpoints/model.pt').read_bytes(), b'previous')

    def test_changed_back_content_still_publishes_new_latest_snapshot(self):
        path = self.put('medium/metrics.json', b'A')
        first = self.store.sync(self.source, 'medium')
        self.assertIsNone(self.store.sync(self.source, 'medium'))
        path.write_bytes(b'B')
        self.store.sync(self.source, 'medium')
        path.write_bytes(b'A')
        latest = self.store.sync(self.source, 'medium')
        self.assertNotEqual(latest, first)
        target = self.root/'restored'
        self.store.restore(target)
        self.assertEqual((target/'medium/metrics.json').read_bytes(), b'A')

    def test_restore_refuses_to_overwrite_different_local_weights(self):
        self.put('easy/checkpoints/model.pt')
        self.store.sync(self.source, 'easy')
        target = self.root/'restored/easy/checkpoints/model.pt'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'unrelated-local-training')
        with self.assertRaises(FileExistsError):
            self.store.restore(self.root/'restored')
        self.assertEqual(target.read_bytes(), b'unrelated-local-training')

    def test_corrupt_remote_blob_cannot_be_promoted_to_checkpoint(self):
        self.put('easy/model.pt')
        self.store.sync(self.source, 'easy')
        model_asset = next(n for n in self.store.content if n.endswith('.pt'))
        self.store.content[model_asset] = b'broken'
        with self.assertRaises(ValueError):
            self.store.restore(self.root/'restored')
        self.assertFalse((self.root/'restored/easy/model.pt').exists())

    def test_path_traversal_is_rejected(self):
        for name in ('../escape', '/absolute', 'C:/escape', 'easy\\..\\escape'):
            with self.assertRaises(ValueError):
                safe_target(self.root, name)

    def test_verified_dataset_cache_does_not_redownload(self):
        archive = self.put('cache/example.zip', b'archive-data')
        manifest = self.put('manifest.json', json.dumps(dict(archives=dict(state=dict(
            filename='example.zip', bytes=archive.stat().st_size, sha256=sha256(archive),
            url='https://github.com/owner/repo/releases/download/data/example.zip')))).encode())
        with patch('urllib.request.urlopen', side_effect=AssertionError('no network')):
            self.assertEqual(download_archive(manifest, cache=archive.parent), archive)


class WorkflowTests(unittest.TestCase):
    def test_notebook_has_separate_levels_and_no_drive_mount(self):
        nb = make_notebook()
        self.assertEqual(len(nb['cells']), 19)
        texts = [''.join(c['source']) for c in nb['cells']]
        for i, cell in enumerate(nb['cells']):
            self.assertEqual(cell['cell_type'], 'code')
            compile(texts[i], f'cell-{i}', 'exec')
        scope = {}
        exec(texts[0], scope)
        self.assertEqual(scope['CFG'], CONFIG)
        self.assertNotIn('drive.mount', '\n'.join(texts))
        self.assertIn('git', texts[1])
        self.assertIn("userdata.get('GH_TOKEN')", texts[2])
        for level in ('easy', 'medium', 'hard'):
            for action in ('smoke', 'train', 'test', 'evaluate'):
                self.assertEqual(sum(f'experiment.{action}("{level}")' in s for s in texts), 1)

    def test_source_bundle_compiles_and_sync_follows_completed_episode_write(self):
        sources = source_bundle()
        for name, source in sources.items():
            compile(source, name, 'exec')
        evaluator = sources['colab_eval_modular.py']
        self.assertEqual(evaluator.count('sync_from_env()'), 1)
        self.assertLess(evaluator.index("save_json(job['output']"), evaluator.index('sync_from_env()'))
        self.assertIn('NextPickObserver', evaluator)

    def test_atomic_checkpoint_is_closed_before_publish_callback(self):
        original = '''def save_ckpt(run_name, tag):
    torch.save({
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')
'''
        updated = atomic_checkpoint_patch(original)
        self.assertLess(updated.index('os.replace'), updated.index('sync_from_env()'))
        self.assertIn("checkpoint_path+'.tmp'", updated)
        with self.assertRaises(ValueError):
            atomic_checkpoint_patch('# upstream changed')

    def test_smoke_disables_remote_and_env_is_restored_after_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(CONFIG)
            cfg.update(profile='smoke', output_root=temp, project_dir=str(Path(__file__).resolve().parents[1]))
            exp = GitHubExperiment(cfg, source_bundle())
            self.assertFalse(exp.remote_enabled)
            with patch.dict(os.environ, {'MOVEBOXES_SYNC_ROOT': 'parent'}, clear=False):
                with self.assertRaises(RuntimeError):
                    with exp.sync_environment('hard'):
                        self.assertNotIn('MOVEBOXES_SYNC_ROOT', os.environ)
                        raise RuntimeError('interruption')
                self.assertEqual(os.environ['MOVEBOXES_SYNC_ROOT'], 'parent')

    def test_training_and_evaluation_seeds_remain_separate(self):
        exp = GitHubExperiment(CONFIG, source_bundle())
        self.assertFalse(set(exp.test_seeds()) & set(exp.seeds(True)))
        self.assertFalse(set(exp.test_seeds()) & set(exp.seeds(False)))
        self.assertFalse(set(exp.seeds(True)) & set(exp.seeds(False)))
        self.assertEqual(exp.train_flags('hard')['batch_size'], 32)
        self.assertEqual(exp.train_flags('hard')['max_episode_steps'], 200)

    def test_later_grasp_weighting_and_regrasp_diagnostics(self):
        signal = [0]*3+[1]*4+[0]+[1]*3+[0]*3+[1]*4+[0]*3+[1]*2
        self.assertEqual(stable_grasp_events(signal), [3, 14])
        weights, audit = sampling_weights([signal], [(0,t-1,t+15) for t in range(20)], 2, before=2, after=1)
        self.assertEqual(weights[0], 1)
        self.assertEqual(weights[14], 3)
        summary = progress_summary([dict(mean_sorted=1, next_pick=dict(stable_grasp_cycles=2, gripper_sign_reversals=4))])
        self.assertEqual(summary['at_least_two_sorted_rate'], 0)
        self.assertEqual(summary['second_cycle_given_first_rate'], 1)


if __name__ == '__main__':
    unittest.main()
