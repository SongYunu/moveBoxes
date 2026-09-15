import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ver2/stages'), str(ROOT/'ver2'), str(ROOT)]
import torch
from stage_chunk_policy import ChunkStagePolicy, load_chunk_stage
from stage_policy import load_stage
from stage_model import StageACT
from stage_schema import PICK, CARRY, PLACE, COMPLETE, RECOVER
from build_stage_deadline_notebook import make_notebook as make_deadline_notebook


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
        self.assertIn("show_official_video('smoke')", deadline_source)
        self.assertIn("show_official_video('default')", deadline_source)
        self.assertIn("show_official_video('public_100ep')", deadline_source)
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


if __name__ == '__main__':
    unittest.main()
