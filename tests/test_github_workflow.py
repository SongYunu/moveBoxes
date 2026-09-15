"""Offline tests; fake transport is used, no real GPU score is asserted."""
import ast
import copy
import json
import os
import tempfile
import unittest
import hashlib
import types
import sys
import io
import contextlib
from pathlib import Path
from unittest.mock import patch

from build_github_notebook import CONFIG, make_notebook
from github_store import AssetUploadError, GitHubStore, safe_target, sha256
from github_data import download_archive
from marso_github import (GitHubExperiment, source_bundle, atomic_checkpoint_patch, final_checkpoint_patch,
                          quiet_training_patch, github_token)
from marso_experiment import save_json, read_json
from colab_eval_modular import restore_rows
from colab_trace import TraceObserver
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

    def test_episode_backup_reuses_checkpoint_bytes_but_publishes_changed_metrics(self):
        model = self.put('easy/checkpoints/model.pt')
        metrics = self.put('easy/metrics.json', b'episode 1')
        self.store.sync(self.source, 'easy', refresh=False)
        with patch('github_store.shutil.copy2', wraps=__import__('shutil').copy2) as copies, \
                patch.object(self.store, 'load', wraps=self.store.load) as loads:
            metrics.write_bytes(b'episode 2 completed')
            self.store.sync(self.source, 'easy', refresh=False)
            self.assertEqual(loads.call_count, 0)
            self.assertEqual([c.args[0] for c in copies.call_args_list], [metrics.resolve()])
        restored = self.root/'cached-restore'
        self.store.restore(restored)
        self.assertEqual((restored/'easy/checkpoints/model.pt').read_bytes(), model.read_bytes())
        self.assertEqual((restored/'easy/metrics.json').read_bytes(), metrics.read_bytes())

    def test_replaced_checkpoint_invalidates_cache(self):
        model = self.put('hard/checkpoints/model.pt', b'old-model')
        self.store.sync(self.source, 'hard', refresh=False)
        replacement = model.with_suffix('.tmp')
        replacement.write_bytes(b'new-model')
        os.replace(replacement, model)
        self.store.sync(self.source, 'hard', refresh=False)
        self.store.restore(self.root/'replaced-restore')
        self.assertEqual((self.root/'replaced-restore/hard/checkpoints/model.pt').read_bytes(), b'new-model')

    def test_unresolved_422_rotates_release_part_and_keeps_uploading(self):
        path = self.put('medium/checkpoints/latest.pt', b'checkpoint')
        store = GitHubStore('owner/repo', 'run-test', 'token')
        store.release = {'id': 1}
        store.part = 1
        attempts = []

        def upload_once(source, name):
            attempts.append((store.part, name))
            if store.part == 1:
                raise AssetUploadError(422)
            return {'name':name, 'size':Path(source).stat().st_size, 'url':name}

        def load(create=True):
            store.release = {'id': 1}
            store.part = 1
            store.assets = {}
            store.write_asset_count = 0
            return True

        def next_part():
            store.part = 2
            store.release = {'id': 2}
            store.write_asset_count = 0

        with patch.object(store, '_upload_once', side_effect=upload_once), \
             patch.object(store, 'load', side_effect=load), \
             patch.object(store, '_next_part', side_effect=next_part), \
             patch('github_store.time.sleep'):
            uploaded = store.upload(path, 'immutable.pt')
        self.assertEqual(uploaded['size'], path.stat().st_size)
        self.assertEqual(attempts, [(1, 'immutable.pt'), (2, 'immutable.pt')])


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
        self.assertIn('experiment.connect()', texts[2])
        for level in ('easy', 'medium', 'hard'):
            for action in ('smoke', 'train', 'test', 'evaluate', 'diagnose'):
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

    def test_existing_environment_token_never_prompts_or_reads_colab_secrets(self):
        with patch.dict(os.environ, {'GH_TOKEN': ' personal-token '}), \
                patch('getpass.getpass', side_effect=AssertionError('unexpected prompt')):
            self.assertEqual(github_token(), 'personal-token')

    def test_missing_secret_falls_back_to_private_prompt(self):
        class MissingSecret(Exception):
            pass
        userdata = types.SimpleNamespace(SecretNotFoundError=MissingSecret, NotebookAccessError=MissingSecret,
                                         get=lambda key: (_ for _ in ()).throw(MissingSecret()))
        colab = types.ModuleType('google.colab')
        colab.userdata = userdata
        with patch.dict(os.environ, {'GH_TOKEN': ''}), \
                patch.dict(sys.modules, {'google.colab': colab}), \
                patch('getpass.getpass', return_value='prompt-token'):
            self.assertEqual(github_token(), 'prompt-token')
            self.assertEqual(os.environ['GH_TOKEN'], 'prompt-token')

    def test_rerun_bootstrap_object_auto_connects_once(self):
        exp = GitHubExperiment(CONFIG, source_bundle())
        def connect():
            exp.connected = True
        with patch.object(exp, 'connect', side_effect=connect) as connection:
            exp._ready()
            exp._ready()
            self.assertEqual(connection.call_count, 1)

    def test_failed_initial_backup_does_not_leave_experiment_connected(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = GitHubExperiment(dict(CONFIG, output_root=temp), source_bundle())
            store = MemoryStore()
            store.repository, store.tag = 'owner/repo', 'run-test'
            def local_connect():
                exp.connected = True
            with patch('marso_github.github_token', return_value='token'), \
                    patch('marso_github.GitHubStore', return_value=store), \
                    patch('marso_github.NextPickExperiment.connect', side_effect=local_connect), \
                    patch.object(exp, 'sync_common', side_effect=RuntimeError('upload failed')):
                with self.assertRaisesRegex(RuntimeError, 'upload failed'):
                    exp.connect()
                self.assertFalse(exp.connected)

    def test_completed_and_partial_state_is_refreshed_before_common_backup(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = dict(CONFIG, output_root=temp)
            exp = GitHubExperiment(cfg, source_bundle())
            exp.connected = True
            exp.session = exp.run_dir/'sessions/test'
            exp.session.mkdir(parents=True)
            ckdir = exp.run_dir/'easy/checkpoints'
            ckdir.mkdir(parents=True)
            (ckdir/'best_eval_sort_accuracy.pt').write_bytes(b'weights')
            save_json(exp.run_dir/'easy/training_complete.json', dict(total_iters=30000))
            job = dict(level='easy', seeds=exp.seeds(False), max_steps=200)
            save_json(exp.run_dir/'easy/metrics.json', dict(complete=False, n_episodes=74,
                sort_accuracy=0.493243, episodes=[dict(seed=s) for s in job['seeds'][:74]], protocol=job))
            exp.store = MemoryStore()
            exp.sync_common()
            restored = Path(temp)/'remote-copy'
            exp.store.restore(restored)
            summary = read_json(restored/'summary.json')
            easy, medium, hard = summary['level_status']
            self.assertEqual(easy['checkpoints'], 1)
            self.assertTrue(easy['train_complete'])
            self.assertIn('74/100', easy['status'])
            self.assertEqual(easy['partial_sort_accuracy'], 0.493243)
            self.assertNotIn('easy', summary['scores'])
            self.assertEqual(medium['status'], '모델 없음')
            self.assertEqual(hard['status'], '모델 없음')

    def test_prior_evaluation_identity_and_74_episode_resume_are_unchanged(self):
        bundle = source_bundle()
        # Actual evaluator and diagnostics digests used by the saved v02 run.
        self.assertEqual(hashlib.sha256(bundle['colab_eval_modular.py'].encode()).hexdigest(),
                         'f8d2ab5168de82c57ef36f37f650f9dfcf1ae207786bac3d5c733e2dba7ebee6')
        self.assertEqual(hashlib.sha256((bundle['next_pick_sampling.py']+bundle['next_pick_diagnostics.py']).encode()).hexdigest(),
                         '0af2e35ccf5fe5c544a05952881318d6ace77585953d436d097f99a7ca8dc123')
        job = dict(fingerprint='saved-job', seeds=list(range(30000, 30100)))
        rows = [dict(seed=s, sort_accuracy=.5, mean_sorted=1, all_placed_rate=0,
                     mean_steps=100, mis_sort_rate=0) for s in job['seeds'][:74]]
        restored = restore_rows(dict(protocol=job, episodes=rows), job)
        self.assertEqual(job['seeds'][len(restored):], list(range(30074, 30100)))
        self.assertEqual(restore_rows(dict(protocol=job, episodes=rows), dict(job, fingerprint='other')), [])

    def test_final_weights_are_saved_before_last_evaluation_even_without_new_best(self):
        source = '''def finish():
    run_name = 'test'
    evaluate_and_save_best(args.total_iters)
'''
        calls = []
        scope = dict(args=types.SimpleNamespace(total_iters=30000),
                     save_ckpt=lambda run, tag: calls.append(('save', tag)),
                     evaluate_and_save_best=lambda iteration: calls.append(('evaluate', iteration)))
        exec(final_checkpoint_patch(source), scope)
        scope['finish']()
        self.assertEqual(calls, [('save', '30000'), ('evaluate', 30000)])
        with self.assertRaises(ValueError):
            final_checkpoint_patch('upstream changed')

    def test_training_progress_is_bounded_in_a_colab_subprocess(self):
        source = '''def train():
    pbar = tqdm(total=args.total_iters)
    for iteration in range(args.total_iters):
        pbar.set_postfix({"loss": total_loss.item()})
'''
        calls = []
        bar = types.SimpleNamespace(set_postfix=lambda values, **kw: calls.append(kw))
        def tqdm(**kwargs):
            self.assertTrue(kwargs['disable'])
            return bar
        scope = dict(args=types.SimpleNamespace(total_iters=100, log_freq=25),
                     tqdm=tqdm, total_loss=types.SimpleNamespace(item=lambda: .1),
                     os=types.SimpleNamespace(isatty=lambda fd: False))
        exec(quiet_training_patch(source), scope)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            scope['train']()
        self.assertEqual(len(output.getvalue().splitlines()), 5)
        self.assertTrue(all(c['refresh'] is False for c in calls))

    def test_trace_copies_states_and_passes_actions_through_without_mutation(self):
        class Tensor:
            def __init__(self, values):
                self.values = values
                self.shape = (len(values), len(values[0])) if isinstance(values[0], list) else (len(values),)
            def __getitem__(self, index):
                if isinstance(index, tuple):
                    return types.SimpleNamespace(item=lambda: self.values[index[0]][index[1]])
                return Tensor(self.values[index])
            def detach(self):
                return self
            def cpu(self):
                return self
            def tolist(self):
                return copy.deepcopy(self.values)
        action = Tensor([[.1, -.2, .3, -1.]])
        state = Tensor([[0.]*54])
        policy = types.SimpleNamespace(reset=lambda: None, act=lambda obs, deterministic=True: action)
        observer = TraceObserver(policy)
        signal = [0]*3+[1]*4+[0]*4+[1]*4
        for grasp in signal:
            state.values[0][25] = grasp
            self.assertIs(observer.act(state), action)
        self.assertEqual([f['state'][25] for f in observer.frames], signal)
        self.assertEqual(observer.report()['stable_grasp_cycles'], 2)
        self.assertTrue(all(f['action'] == [.1, -.2, .3, -1.] for f in observer.frames))
        observer.reset()
        self.assertEqual(observer.frames, [])
        self.assertEqual(observer.report()['observed_steps'], 0)


if __name__ == '__main__':
    unittest.main()
