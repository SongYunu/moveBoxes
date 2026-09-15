import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ver2/stages'), str(ROOT/'ver2'), str(ROOT)]
import torch
from stage_chunk_policy import ChunkStagePolicy, load_chunk_stage
from stage_policy import load_stage
from stage_model import StageACT
from stage_schema import PICK, CARRY, PLACE, COMPLETE, RECOVER
from stage_compare import evaluate, summarize, variant_config, load_spec
from stage_compare_restore import restore_baselines
from build_stage_compare_notebook import make_notebook


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
        for frame in (torch.zeros(54), torch.zeros(1, 26), torch.full((1,54), float('nan'))):
            with self.assertRaises(ValueError):
                ChunkStagePolicy(Controlled()).act(frame)

    def test_loader_baseline_parity_and_three_state_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            for dim in (54, 72, 90):
                path, base = checkpoint(directory, dim)
                space = SimpleNamespace(shape=(4,))
                old = load_stage(path, torch.zeros(1, dim), space, 'cpu', **base)
                a = load_chunk_stage(path, torch.zeros(1, dim), space, 'cpu', **variant_config(base, 'A', {}))
                for _ in range(9):
                    frame = torch.randn(1, dim)
                    torch.testing.assert_close(a.act(frame), old.act(frame), atol=0, rtol=0)
                for variant in ('B', 'C'):
                    policy = load_chunk_stage(path, frame, space, 'cpu', **variant_config(base, variant, {}))
                    action = policy.act(frame)
                    self.assertEqual(action.shape, (1, 4))
                    self.assertTrue(torch.isfinite(action).all() and action.abs().max() <= 1)
                wrong = copy.deepcopy(base)
                wrong['model_config']['history'] = 3
                with self.assertRaises(ValueError):
                    load_chunk_stage(path, frame, space, 'cpu', **wrong)

    def test_wrong_checkpoint_family_and_missing_baseline_settings_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, base = checkpoint(directory)
            with self.assertRaises(ValueError):
                load_spec(dict(checkpoint=str(path), policy_config={}), 54)
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                load_spec(dict(checkpoint=str(path), policy_config=base, checkpoint_sha256='bad'), 54)
            saved = torch.load(path, weights_only=True)
            saved['format'] = 'moveboxes-hard-target-act-v2'
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                load_spec(dict(checkpoint=str(path), policy_config=base), 54)

    def test_summary_does_not_invent_missing_level_score(self):
        with tempfile.TemporaryDirectory() as directory:
            results = {'easy': {'A': {'sort_accuracy': 1.}}}
            table = summarize(results, directory)
            self.assertIsNone(table['A']['overall'])
            for level, score in (('medium', .4), ('hard', .1)):
                results[level] = {'A': {'sort_accuracy': score}}
            table = summarize(results, directory)
            self.assertAlmostEqual(table['A']['overall'], .37)

    def test_evaluation_resets_every_episode_and_resumes_without_repeating(self):
        class Env:
            single_action_space = SimpleNamespace(shape=(4,))
            def reset(self, seed):
                return torch.zeros(1, 54), {}
            def close(self):
                pass
        calls = []
        def rollout(env, agent, device, n, seeds, max_steps):
            self.assertEqual(agent.policy.step, 0)
            self.assertEqual(n, 1)
            calls.append(seeds[0])
            for _ in range(4):
                agent.act(env.reset(0)[0])
            return dict(sort_accuracy=.5, all_placed_rate=0, mean_sorted=1, mean_steps=200, mis_sort_rate=0)
        utils = ModuleType('warehouse_sort.utils')
        utils.__file__ = __file__
        utils.compose_cfg = lambda args: SimpleNamespace(randomization={})
        utils.make_env = lambda *a, **kw: (Env(), False)
        utils.rollout_metrics = rollout
        env_module = ModuleType('warehouse_sort.env')
        env_module.__file__ = __file__
        package = ModuleType('warehouse_sort')
        package.utils, package.env = utils, env_module
        omega = SimpleNamespace(OmegaConf=SimpleNamespace(to_container=lambda *a, **kw: {}))
        with tempfile.TemporaryDirectory() as directory:
            path, base = checkpoint(directory)
            config = dict(output=str(Path(directory)/'results'), seeds=[61000,61001], max_steps=200,
                levels=dict(easy=dict(checkpoint=str(path), policy_config=base)))
            with patch.dict(sys.modules, {'warehouse_sort': package, 'warehouse_sort.utils':utils,
                                         'warehouse_sort.env':env_module, 'omegaconf':omega}):
                evaluate(config, 'cpu')
                self.assertEqual(calls, [61000,61001]*3)
                evaluate(config, 'cpu')
                self.assertEqual(len(calls), 6)
                config['seeds'] = [61003]
                with self.assertRaisesRegex(ValueError, 'different inputs'):
                    evaluate(config, 'cpu')

    def test_notebook_cells_compile_and_only_evaluate(self):
        notebook = make_notebook()
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
        self.assertIn('stage_compare.py', source)
        self.assertNotIn('experiment.train(', source)
        self.assertNotIn('experiment.collect(', source)
        self.assertNotIn('pip\', \'install\', \'--upgrade', source)

    def test_restore_is_read_only_hash_pinned_and_never_overwrites_existing_checkpoint(self):
        import hashlib
        content = b'checkpoint-fixture'
        sha = hashlib.sha256(content).hexdigest()
        baseline = dict(easy=dict(run='fixture', checkpoint='best.pt', sha256=sha))
        calls = []
        class Store:
            def __init__(self, *args, **kwargs):
                self.assets = {'snapshot-easy-001.json': {'kind':'snapshot'},
                               'weights': {'kind':'weights', 'size':len(content)}}
            def load(self, create):
                calls.append(('load', create))
                return True
            def download(self, asset, path):
                calls.append(('download', asset['kind']))
                if asset['kind'] == 'weights':
                    path.write_bytes(content)
                else:
                    path.write_text(json.dumps(dict(version=1, scope='easy', entries=[dict(
                        sha256=sha, path='easy/checkpoints/best.pt', asset='weights', bytes=len(content))])))
        with tempfile.TemporaryDirectory() as directory:
            with patch('stage_compare_restore.BASELINES', baseline), patch('stage_compare_restore.GitHubStore', Store):
                specs = restore_baselines(directory, ['easy'])
                self.assertEqual(specs['easy']['checkpoint_sha256'], sha)
                self.assertIn(('load', False), calls)
                count = len(calls)
                restore_baselines(directory, ['easy'])
                self.assertEqual(len(calls), count)
                Path(specs['easy']['checkpoint']).write_bytes(b'other')
                with self.assertRaises(FileExistsError):
                    restore_baselines(directory, ['easy'])
                self.assertEqual(Path(specs['easy']['checkpoint']).read_bytes(), b'other')


if __name__ == '__main__':
    unittest.main()
