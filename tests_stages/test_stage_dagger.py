import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT/'ver2/stages'), str(ROOT/'ver2'), str(ROOT)]

from marso_experiment import digest
from build_stage_dagger_notebook import make_notebook
from stage_dagger import FROZEN_MODULES, materialize_training_manifest
from stage_dagger_collect import collect_episode, select_executed
from stage_model import StageACT
from stage_train import train


def state():
    obs = np.zeros(54, dtype=np.float32)
    obs[18:21] = [.2, 0, .22]
    obs[26:29] = [.2, 0, .03]
    obs[33:36] = [.2, .12, .03]
    obs[40:44] = [1, 0, 0, 1]
    obs[44:50] = [0, -.36, 0, 0, .36, 0]
    return obs


class DaggerTests(unittest.TestCase):
    def test_colab_uses_official_eval_and_never_packages_teacher(self):
        notebook = make_notebook()
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        self.assertNotIn('drive.mount', source)
        self.assertIn('DAGGER_ROUNDS = 4', source)
        self.assertIn('EPISODES_PER_ROUND = 24', source)
        self.assertIn('TRAIN_ITERS_PER_ROUND = 2000', source)
        self.assertIn("UPSTREAM/'eval.py'", source)
        self.assertIn('MIN_EXTRA_SORTED = 2', source)
        self.assertIn("manifest['levels'][LEVEL]['selection'] = 'supervised_dagger'", source)
        self.assertNotIn('prepare_success_rl(', source)
        self.assertNotIn('prepare_pick_residual_rl(', source)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['metadata'].get('id','cell'), 'exec')

    def test_action_mixing_and_failed_rollout_labels_are_retained(self):
        rng = np.random.default_rng(4)
        expert = np.asarray([1, .5, 0, -1], dtype=np.float32)
        policy = np.asarray([-1, 0, .25, 1], dtype=np.float32)
        np.testing.assert_array_equal(select_executed(expert, policy, 1, rng)[0], expert)
        np.testing.assert_array_equal(select_executed(expert, policy, 0, rng)[0], policy)

        class Policy:
            def reset(self): pass
            def act(self, obs, deterministic=True):
                return torch.tensor([[-1., 0., 0., 1.]])
        class Env:
            def __init__(self):
                self.unwrapped, self.num_parcels, self.parcels = self, 2, [0, 1]
                self.agent = SimpleNamespace(is_grasping=lambda parcel:torch.tensor([False]))
            def observation(self):
                value = state(); value[0] = self.t
                return torch.from_numpy(value[None])
            def reset(self, seed):
                self.t = 0
                return self.observation(), {}
            def step(self, action):
                self.t += 1
                return self.observation(), torch.zeros(1), torch.tensor([False]), torch.tensor([False]), {}
            def evaluate(self):
                return dict(success_count=torch.tensor([0]))
        arrays, report = collect_episode(Env(), Policy(), 10,
            dict(max_steps=5, beta=0.))
        self.assertTrue(report['accepted'])
        self.assertFalse(report['success'])
        self.assertEqual(report['policy_steps'], 5)
        self.assertEqual(arrays['obs'].shape, (6, 54))
        self.assertTrue(arrays['perturbed'].all())
        self.assertEqual(arrays['actions'].shape, (5, 4))

    def test_training_manifest_combines_clean_and_all_dagger_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = SimpleNamespace(run_dir=root/'run')
            clean = base.run_dir/'medium/collection'; clean.mkdir(parents=True)
            dagger = base.run_dir/'medium/dagger_v1'
            def episode(folder, name):
                path = folder/name
                np.savez_compressed(path, obs=np.zeros((2,72),np.float32),
                    actions=np.zeros((1,4),np.float32), previous_stage=np.zeros(1,np.int64),
                    stage=np.zeros(1,np.int64), gate=np.zeros(1,np.int64),
                    target=np.zeros(1,np.int64), perturbed=np.zeros(1,bool))
                return dict(file=name, sha256=digest(path), accepted=True,
                            valid_for_training=True, success=False)
            clean_entry = episode(clean, 'clean.npz')
            (clean/'manifest.json').write_text(json.dumps(dict(
                complete=True, episodes=[clean_entry])))
            for number in (1, 2):
                folder = dagger/f'round_{number:02d}/collection'; folder.mkdir(parents=True)
                entry = episode(folder, f'r{number}.npz')
                (folder/'manifest.json').write_text(json.dumps(dict(
                    kind='dagger-v1', complete=True, episodes=[entry])))
            result = materialize_training_manifest(base, 'medium', dagger, 2)
            manifest = json.loads(result.read_text())
            self.assertEqual(manifest['clean_episodes'], 1)
            self.assertEqual(manifest['dagger_episodes'], 2)
            self.assertEqual(len(manifest['episodes']), 3)
            self.assertTrue(all((result.parent/row['file']).is_file()
                                for row in manifest['episodes']))

    def test_frozen_encoder_remains_exact_during_supervised_correction(self):
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root/'data.h5'
            with h5py.File(data, 'w') as handle:
                for index in range(4):
                    group = handle.create_group(f'traj_{index}')
                    obs = np.repeat(state()[None], 7, axis=0)
                    obs[:, 0] = index
                    actions = np.zeros((6,4), dtype=np.float32); actions[:,3] = 1
                    group.create_dataset('obs', data=obs)
                    group.create_dataset('actions', data=actions)
            arch = dict(state_dim=54, history=4, chunk_size=4, width=32,
                        heads=4, layers=1, latent_dim=4)
            initial_model = StageACT(arch)
            initial = root/'initial.pt'
            torch.save(dict(format='moveboxes-stage-act-v1', model_config=arch,
                            model=initial_model.state_dict(), step=10), initial)
            cfg = dict(seed=9, batch_size=4, lr=1e-3, total_iters=3, save_freq=3,
                warmup_steps=1, kl_weight=.001, stage_loss_weight=.3, gate_loss_weight=.3,
                position_noise=0., validation_batches=1, amp=False,
                console_interval_seconds=999, action_training_mode='prior',
                freeze_modules=['obs_proj','encoder'])
            job = dict(folder=str(root/'train'), data=str(data), model_config=arch,
                train_config=cfg, policy_config=dict(model_config=arch), num_demos=None,
                warm_start=str(initial), source_sha256='dagger-test', device='cpu')
            with patch('stage_train.sync_from_env'):
                train(job)
            trained = torch.load(root/'train/checkpoints/latest.pt', weights_only=True)['model']
            before = initial_model.state_dict()
            for name in trained:
                if name.startswith(('obs_proj.', 'encoder.')):
                    torch.testing.assert_close(trained[name], before[name], atol=0, rtol=0)
            self.assertTrue(any(not torch.equal(trained[name], before[name])
                                for name in trained if name.startswith('output.')))
            self.assertIn('encoder', FROZEN_MODULES)


if __name__ == '__main__':
    unittest.main()
