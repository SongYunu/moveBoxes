import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ver2/stages'), str(ROOT/'ver2'), str(ROOT)]
import torch
from stage_chunk_policy import ChunkStagePolicy, load_chunk_stage
from stage_policy import load_stage
from stage_model import StageACT
from stage_schema import PICK, CARRY, PLACE, COMPLETE, RECOVER
from build_stage_deadline_notebook import make_notebook as make_deadline_notebook
from build_stage_deadline_notebook import CONFIG as DEADLINE_CONFIG
from build_stage_deadline_notebook import make_runtime_repair_notebook
from stage_experiment import StageExperiment
from stage_pick_diagnose import decision_row
from stage_pick_finetune import prepare_pick_finetune


class Controlled(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.cfg = dict(state_dim=54, history=2, chunk_size=8)
        self.decoded_batches = []

    def encode(self, obs, previous):
        phase = obs.new_full((len(obs), 4), -20)
        gate = obs.new_full((len(obs), 3), -20)
        phase.scatter_(1, obs[:, -1, 0:1].long(), 20)
        gate.scatter_(1, obs[:, -1, 1:2].long(), 20)
        gate[obs[:, -1, 2] == 1] = 0
        return obs, obs, phase, gate

    def decode(self, x, memory, stage):
        self.decoded_batches.append(len(x))
        actions = x.new_zeros(len(x), 8, 4)
        actions[:, :, 0] = x[:, -1, 3:4] + torch.arange(8)*.01
        actions[:, :, 3] = x[:, -1, 4:5]
        return actions, None


def obs(batch=1, phase=PICK, gate=0, marker=.1, grip=1.):
    x = torch.zeros(batch, 54)
    x[:, 0], x[:, 1], x[:, 3], x[:, 4] = phase, gate, marker, grip
    return x


def checkpoint(directory, dim=54):
    arch = dict(state_dim=dim, history=2, chunk_size=8, width=16, heads=2, layers=1, latent_dim=4)
    model = StageACT(arch)
    path = Path(directory)/'best.pt'
    torch.save(dict(format='moveboxes-stage-act-v1', model_config=arch, model=model.state_dict()), path)
    base = dict(model_config=arch, ensemble_window=4, temporal_decay=.25, gate_threshold=.65, stage_threshold=.6)
    return path, base


class ChunkTests(unittest.TestCase):
    def test_pick_trace_records_xyz_without_changing_observation_or_action(self):
        state = torch.zeros(1,54)
        state[0,18:21] = torch.tensor([.1,.2,.22])
        state[0,26:29] = torch.tensor([.11,.21,.026])
        state[0,33:36] = torch.tensor([.4,.4,.026])
        action = torch.tensor([[.1,-.1,-.2,1.]])
        initial = state.clone(), action.clone()
        row = decision_row(10, state, action, dict(stage=torch.tensor([PICK]), accepted=torch.tensor([False])))
        self.assertEqual(row['nearest_parcel_xy'],0)
        self.assertAlmostEqual(row['tcp_minus_nearest_parcel_z'],.194,places=5)
        self.assertEqual(row['stage'], PICK)
        torch.testing.assert_close(state, initial[0],rtol=0,atol=0)
        torch.testing.assert_close(action, initial[1],rtol=0,atol=0)

    def test_pick_finetune_prepares_separate_run_from_exact_packaged_weights(self):
        import contextlib, hashlib
        from marso_experiment import digest

        class OfflineExperiment:
            def __init__(self,cfg,sources):
                self.cfg,self.sources = cfg,sources
                self.run_dir = Path(cfg['output_root'])/(cfg['run_name']+'_benchmark')
            def connect(self):
                self.run_dir.mkdir(parents=True,exist_ok=True)
            def _stage_helpers(self):
                pass
            def persist_operation(self,level):
                return contextlib.nullcontext()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ('act_v2_model.py','act_v2_data.py','stage_schema.py','stage_labels.py',
                     'stage_model.py','stage_data.py','stage_train.py')
            sources = dict.fromkeys(names,'saved-source')
            parent = OfflineExperiment(dict(output_root=str(root),run_name='protected'),sources)
            candidate = parent.run_dir/'integrated_candidate'
            ckdir = candidate/'checkpoints/easy'
            ckdir.mkdir(parents=True)
            path,policy = checkpoint(ckdir,54)
            path.rename(ckdir/'model.pt')
            policy.update(stage_horizons=dict(pick=2,carry=6,place=2,done=1),stage_aware_chunk=True)
            (ckdir/'policy_config.json').write_text(json.dumps(policy))
            (candidate/'manifest.json').write_text(json.dumps(dict(levels=dict(easy=dict(checkpoint_sha256=digest(ckdir/'model.pt'))))))
            collection = parent.run_dir/'easy/collection'
            collection.mkdir(parents=True)
            episode = collection/'episode.npz'
            episode.write_bytes(b'saved-recovery')
            (collection/'manifest.json').write_text(json.dumps(dict(complete=True,episodes=[dict(file=episode.name,sha256=digest(episode))])))
            (parent.run_dir/'easy/stage_train_job.json').write_text(json.dumps(dict(
                model_config=policy['model_config'],train_config=dict(lr=.0004,total_iters=12000),
                num_demos=200,data='original-dataset.h5',
                source_sha256=hashlib.sha256(''.join(sources[n] for n in names).encode()).hexdigest())))
            preserved = {p:p.read_bytes() for p in parent.run_dir.rglob('*') if p.is_file()}
            child = prepare_pick_finetune(parent,candidate)
            job = json.loads((child.run_dir/'easy/stage_train_job.json').read_text())
            self.assertNotEqual(child.run_dir,parent.run_dir)
            self.assertEqual(job['train_config']['total_iters'],2000)
            self.assertAlmostEqual(job['train_config']['lr'],.00008)
            self.assertEqual(job['train_config']['first_pick_fraction'],.5)
            self.assertEqual(job['policy_config']['stage_horizons'],policy['stage_horizons'])
            self.assertEqual((child.run_dir/'easy/initial_model.pt').read_bytes(),(ckdir/'model.pt').read_bytes())
            self.assertEqual(prepare_pick_finetune(parent,candidate).run_dir,child.run_dir)
            for path, content in preserved.items():
                self.assertEqual(path.read_bytes(),content)

    def setUp(self):
        torch.set_num_threads(2)

    def test_sequential_chunk_and_replan_uses_latest_history(self):
        model = Controlled()
        policy = ChunkStagePolicy(model)
        self.assertAlmostEqual(float(policy.act(obs())[0, 0]), .1, places=6)
        self.assertAlmostEqual(float(policy.act(obs(marker=.5))[0, 0]), .11, places=6)
        self.assertEqual(policy.decoder_calls, 1)
        self.assertAlmostEqual(float(policy.act(obs(marker=.8))[0, 0]), .8, places=6)
        self.assertEqual(policy.decoder_calls, 2)
        self.assertEqual(len(policy.history), 2)

    def test_transition_and_same_stage_recovery_reset_only_affected_environment(self):
        model = Controlled()
        policy = ChunkStagePolicy(model)
        policy.act(obs(batch=2))
        frame = obs(batch=2, marker=.7)
        frame[0, 0], frame[0, 1] = CARRY, COMPLETE
        action = policy.act(frame)
        torch.testing.assert_close(action[:, 0], torch.tensor([.7, .11]))
        self.assertEqual(model.decoded_batches, [2, 1])
        self.assertEqual(policy.remaining.tolist(), [5, 0])
        frame[0, 1], frame[0, 3] = RECOVER, .9
        policy.act(frame)
        self.assertEqual(policy.generation.tolist(), [2, 0])
        self.assertEqual(policy.remaining.tolist(), [5, 1])
        self.assertAlmostEqual(float(policy.action_buffer[0, 0, 0]), .9, places=6)

    def test_uncertain_gate_preserves_stage_and_buffer(self):
        policy = ChunkStagePolicy(Controlled())
        policy.act(obs())
        frame = obs(phase=CARRY, gate=COMPLETE, marker=.8)
        frame[:, 2] = 1
        self.assertAlmostEqual(float(policy.act(frame)[0, 0]), .11, places=6)
        self.assertEqual(int(policy.stage[0]), PICK)
        self.assertEqual(policy.decoder_calls, 1)

    def test_reset_and_batch_resize_clear_buffer_and_filter(self):
        policy = ChunkStagePolicy(Controlled(), gripper_fsm=True)
        first = policy.act(obs())
        policy.act(obs(grip=-1))
        policy.reset()
        self.assertIsNone(policy.action_buffer)
        self.assertIsNone(policy.grip_state)
        self.assertIsNone(policy.last_decision)
        self.assertFalse(policy.history)
        torch.testing.assert_close(policy.act(obs()), first)
        self.assertEqual(policy.act(obs(batch=2)).shape, (2, 4))
        self.assertEqual(policy.decoder_calls, 1)

    def test_official_fixed_episode_boundary_resets_all_runtime_state(self):
        policy = ChunkStagePolicy(Controlled(), gripper_fsm=True, auto_reset_steps=2)
        first = policy.act(obs(marker=.1, grip=1))
        policy.act(obs(marker=.8, grip=-1))
        self.assertEqual(policy.step, 2)
        self.assertTrue(policy.history)
        repeated = policy.act(obs(marker=.1, grip=1))
        torch.testing.assert_close(repeated, first)
        self.assertEqual(policy.step, 1)
        self.assertEqual(policy.decoder_calls, 1)
        self.assertEqual(policy.generation.tolist(), [0])
        self.assertEqual(policy.stage.tolist(), [PICK])
        self.assertEqual(policy.grip_state.tolist(), [1.])

    def test_gripper_filter_follows_confident_learned_commands_and_resets_on_transition(self):
        policy = ChunkStagePolicy(Controlled(), stage_horizons={s:1 for s in ('pick','carry','place','done')}, gripper_fsm=True)
        sequence = [1, -.2, -1, .1, -1, -1, 1]
        self.assertEqual([int(policy.act(obs(grip=g))[0, 3]) for g in sequence], [1, 1, 1, 1, 1, -1, -1])
        self.assertEqual(int(policy.act(obs(phase=CARRY, gate=COMPLETE, grip=1))[0, 3]), 1)
        self.assertEqual(int(policy.act(obs(phase=CARRY, gate=RECOVER, grip=-1))[0, 3]), -1)

    def test_invalid_horizons_and_observation_fail_early(self):
        for value in (0, 9, 1.5, True):
            with self.assertRaises(ValueError):
                ChunkStagePolicy(Controlled(), stage_horizons=dict(pick=value,carry=6,place=2,done=1))
        with self.assertRaises(ValueError):
            ChunkStagePolicy(Controlled(), stage_horizons=dict(pick=2))
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                ChunkStagePolicy(Controlled(), auto_reset_steps=value)
        for frame in (torch.zeros(54), torch.zeros(1, 26), torch.full((1,54), float('nan'))):
            with self.assertRaises(ValueError):
                ChunkStagePolicy(Controlled()).act(frame)

    def test_loader_legacy_parity_and_integrated_policy_all_state_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            for dim in (54, 72, 90):
                path, base = checkpoint(directory, dim)
                space = SimpleNamespace(shape=(4,))
                old = load_stage(path, torch.zeros(1, dim), space, 'cpu', **base)
                a = load_chunk_stage(path, torch.zeros(1, dim), space, 'cpu', **base)
                for _ in range(9):
                    frame = torch.randn(1, dim)
                    torch.testing.assert_close(a.act(frame), old.act(frame), atol=0, rtol=0)
                integrated = copy.deepcopy(base)
                integrated.update(stage_aware_chunk=True,gripper_fsm=True,
                                  stage_horizons=dict(pick=2,carry=6,place=2,done=1),
                                  gripper_margin=.5,gripper_confirm_steps=2,auto_reset_steps=199)
                policy = load_chunk_stage(path, frame, space, 'cpu', **integrated)
                action = policy.act(frame)
                self.assertEqual(action.shape, (1, 4))
                self.assertTrue(torch.isfinite(action).all() and action.abs().max() <= 1)
                wrong = copy.deepcopy(base)
                wrong['model_config']['history'] = 3
                with self.assertRaises(ValueError):
                    load_chunk_stage(path, frame, space, 'cpu', **wrong)

    def test_wrong_checkpoint_family_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, base = checkpoint(directory)
            saved = torch.load(path, weights_only=True)
            saved['format'] = 'moveboxes-hard-target-act-v2'
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                load_chunk_stage(path, torch.zeros(1,54), SimpleNamespace(shape=(4,)), 'cpu', **base)

    def test_candidate_cell_executes_without_imports_from_previous_cells(self):
        notebook = make_deadline_notebook()
        cell = next(c for c in notebook['cells'] if c['metadata'].get('id') == 'integrated-candidate')
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(DEADLINE_CONFIG, output_root=directory, run_name='candidate-test',
                       project_commit='test-project', team='test-team')
            experiment = StageExperiment(cfg, {})
            self.assertEqual(experiment.run_dir.name, 'candidate-test_benchmark')
            original = {}
            for level, dim in (('easy',54), ('medium',72), ('hard',90)):
                folder = experiment.run_dir/level/'checkpoints'
                folder.mkdir(parents=True)
                path, policy = checkpoint(folder, dim)
                latest = folder/'latest.pt'
                path.rename(latest)
                (folder/'policy_config.json').write_text(json.dumps(policy), encoding='utf-8')
                original[level] = latest.read_bytes()
            namespace = dict(CFG=cfg, PROJECT=ROOT, experiment=experiment)
            exec(compile(''.join(cell['source']), 'integrated-candidate', 'exec'), namespace)
            self.assertEqual(namespace['RUN_DIR'], experiment.run_dir)
            manifest = json.loads((namespace['CANDIDATE']/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(set(manifest['levels']), {'easy','medium','hard'})
            for level, row in manifest['levels'].items():
                source = namespace['selected'][level]
                self.assertEqual(source.name, 'latest.pt')
                self.assertEqual(source.read_bytes(), original[level])
                target = namespace['CANDIDATE']/'checkpoints'/level/'model.pt'
                policy = load_chunk_stage(target, torch.zeros(1,row['model_config']['state_dim']),
                    SimpleNamespace(shape=(4,)), 'cpu', **row['policy_config'])
                self.assertEqual(policy.act(torch.zeros(1,row['model_config']['state_dim'])).shape, (1,4))

    def test_notebook_cells_compile_train_resume_and_official_eval(self):
        deadline = make_deadline_notebook()
        for cell in deadline['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell.get('id', cell['metadata'].get('id','cell')), 'exec')
        deadline_source = '\n'.join(''.join(c['source']) for c in deadline['cells'])
        self.assertIn("run_name='moveboxes_stage_chunk_deadline_v1'", Path(ROOT/'build_stage_deadline_notebook.py').read_text(encoding='utf-8'))
        self.assertIn("'output_root': '/content/moveboxes_runs'", deadline_source)
        self.assertIn('experiment.connect()', deadline_source)
        self.assertIn('GH_TOKEN', deadline_source)
        self.assertIn("'save_freq': 1000", deadline_source)
        self.assertIn("'download_cache': '/content/moveboxes_data_cache'", deadline_source)
        self.assertIn('experiment.prepare_data()', deadline_source)
        self.assertIn("experiment.collect('easy')", deadline_source)
        self.assertIn("experiment.train('easy')", deadline_source)
        self.assertIn("CHECKPOINT_OVERRIDES = {'easy':'', 'medium':'', 'hard':''}", deadline_source)
        self.assertIn("checkpoint = folder/'latest.pt'", deadline_source)
        self.assertIn('BENCHMARK_EPISODES', deadline_source)
        self.assertIn("run_official(level, BENCHMARK_CONFIG, 'public_100ep')", deadline_source)
        self.assertIn('from IPython.display import Video, display', deadline_source)
        self.assertIn("show_all_official_videos('smoke')", deadline_source)
        self.assertIn("show_all_official_videos('default')", deadline_source)
        self.assertIn("show_all_official_videos('public_100ep')", deadline_source)
        self.assertIn('BACKUP_EVAL_RESULTS = False', deadline_source)
        self.assertIn('DOWNLOAD_VIDEO = False', deadline_source)
        self.assertIn('SMOKE_SEED', deadline_source)
        self.assertIn('stage_chunk_policy:load_policy', deadline_source)
        self.assertIn("UPSTREAM/'eval.py'", deadline_source)
        self.assertIn("UPSTREAM/'conf/eval/default.yaml'", deadline_source)
        self.assertIn('stage_aware_chunk=True', deadline_source)
        self.assertIn('gripper_fsm=True', deadline_source)
        self.assertIn('auto_reset_steps=MAX_STEPS-1', deadline_source)
        self.assertNotIn('drive.mount', deadline_source)
        self.assertNotIn('restore_baselines', deadline_source)
        self.assertNotIn('comparison.json', deadline_source)
        self.assertNotIn('MANUAL_SELECTION', deadline_source)

    def test_eval_finishes_all_levels_without_calling_github_backup(self):
        notebook = make_deadline_notebook()
        cells = {c['metadata'].get('id'): c for c in notebook['cells']}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root/'official'
            official.mkdir()
            (official/'eval.py').write_text(
                "import sys\nfrom pathlib import Path\n"
                "output=Path(next(a.split('=',1)[1] for a in sys.argv if a.startswith('hydra.run.dir=')))\n"
                "(output/'videos').mkdir(parents=True,exist_ok=True)\n"
                "(output/'videos'/'rollout.mp4').write_bytes(b'test-video')\n"
                "print('[eval] saved rollout video')\n", encoding='utf-8')
            def failing_backup(level):
                raise RuntimeError('GitHub asset upload failed (422)')
            namespace = dict(RUN_DIR=root, CFG={'repo_dir':str(official)},
                CANDIDATE=root/'candidate', selected={'easy':None,'medium':None,'hard':None},
                MAX_STEPS=200, SMOKE_SEED=61000, Path=Path,
                experiment=SimpleNamespace(sync_level=failing_backup))
            exec(''.join(cells['official-smoke']['source']), namespace)
            for level in namespace['selected']:
                self.assertTrue((root/level/'integrated_official_eval/smoke/videos/rollout.mp4').is_file())
            backup = ''.join(cells['optional-eval-backup']['source'])
            exec(backup.replace('BACKUP_EVAL_RESULTS = False','BACKUP_EVAL_RESULTS = True'), namespace)

    def test_video_player_supports_old_colab_constructor(self):
        cells = {c['metadata'].get('id'): c for c in make_deadline_notebook()['cells']}
        displayed = []

        def legacy_video(data=None, **kwargs):
            # Reproduce Colab's os.path.exists(data) before handling filename.
            self.assertTrue(Path(data).exists())
            self.assertNotIn('filename', kwargs)
            self.assertTrue(kwargs['embed'])
            return data

        display_module = SimpleNamespace(Video=legacy_video, display=displayed.append)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for level in ('easy', 'medium', 'hard'):
                folder = root/level/'integrated_official_eval/smoke/videos'
                folder.mkdir(parents=True)
                (folder/'rollout.mp4').write_bytes(b'video')
            namespace = dict(RUN_DIR=root, selected=dict.fromkeys(('easy','medium','hard')))
            with mock.patch.dict(sys.modules, {'IPython.display':display_module}):
                exec(''.join(cells['video-player']['source']), namespace)
                namespace['show_all_official_videos']('smoke')
            self.assertEqual(len(displayed), 3)

    def test_stop_button_terminates_real_eval_subprocess(self):
        import subprocess
        cells = {c['metadata'].get('id'): c for c in make_deadline_notebook()['cells']}
        processes = []
        real_popen = subprocess.Popen

        def tracked_popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        def stop_button(*args, **kwargs):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root/'official'
            official.mkdir()
            (official/'eval.py').write_text(
                "import time\nprint('running evaluation', flush=True)\ntime.sleep(60)\n",
                encoding='utf-8')
            namespace = dict(RUN_DIR=root, CFG={'repo_dir':str(official)},
                CANDIDATE=root/'candidate', selected={'easy':None},
                MAX_STEPS=200, SMOKE_SEED=61000, Path=Path, print=stop_button)
            with mock.patch('subprocess.Popen', side_effect=tracked_popen):
                with self.assertRaises(KeyboardInterrupt):
                    exec(''.join(cells['official-smoke']['source']), namespace)
            self.assertEqual(len(processes), 1)
            self.assertIsNotNone(processes[0].poll())
            self.assertTrue(processes[0].stdout.closed)
            log = root/'easy/integrated_official_eval/smoke/official_eval.log'
            self.assertIn('running evaluation', log.read_text())

    def test_active_runtime_repair_preserves_candidate_and_renders_missing_only(self):
        notebook = make_runtime_repair_notebook()
        cells = {c['metadata']['id']: c for c in notebook['cells']}
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['metadata']['id'], 'exec')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root/'official'
            official.mkdir()
            (official/'eval.py').write_text(
                "import sys\nfrom pathlib import Path\n"
                "output=Path(next(a.split('=',1)[1] for a in sys.argv if a.startswith('hydra.run.dir=')))\n"
                "(output/'videos').mkdir(parents=True,exist_ok=True)\n"
                "(output/'videos'/'rollout.mp4').write_bytes(b'new-video')\n"
                "print('official evaluator completed')\n", encoding='utf-8')
            candidate = root/'integrated_candidate'
            preserved = {}
            for level in ('easy','medium','hard'):
                folder = candidate/'checkpoints'/level
                folder.mkdir(parents=True)
                for name in ('model.pt','policy_config.json'):
                    path = folder/name
                    path.write_bytes(b'existing-'+level.encode())
                    preserved[path] = path.read_bytes()
            (candidate/'manifest.json').write_text(json.dumps(dict(levels=dict.fromkeys(('easy','medium','hard')))))
            existing = root/'easy/integrated_official_eval/smoke/videos/rollout.mp4'
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b'old-video')
            namespace = dict(experiment=SimpleNamespace(run_dir=root), CFG={'repo_dir':str(official)},
                             show_all_official_videos=lambda label:None)
            exec(''.join(cells['runtime-repair-setup']['source']), namespace)
            exec(''.join(cells['runtime-repair-render-missing']['source']), namespace)
            self.assertEqual(existing.read_bytes(), b'old-video')
            for level in ('medium','hard'):
                self.assertTrue((root/level/'integrated_official_eval/smoke/videos/rollout.mp4').is_file())
            for path, content in preserved.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
