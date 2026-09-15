import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ver2/stages'), str(ROOT/'ver2'), str(ROOT)]

import torch
from build_stage_anchor_notebook import make_notebook
from build_stage_pick_residual_notebook import make_notebook as make_residual_notebook
from stage_anchor_continue import ANCHORS, DIMS, package
from stage_model import PickResidualStageACT, StageACT
from stage_pick_residual_rl import group_relative_advantages, group_update, trailing_pick_mask
from stage_success_rl import bernoulli_kl_from_logits, discounted_returns, sparse_success_delta


class AnchorTests(unittest.TestCase):
    def test_pick_residual_notebook_is_restartable_and_uses_robust_selection(self):
        notebook = make_residual_notebook()
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        self.assertNotIn('drive.mount', source)
        self.assertIn("prepare_pick_residual_rl(\n    anchor_exp, 'medium'", source)
        self.assertIn('group_size=4', source)
        self.assertIn('pick_credit_steps=12', source)
        self.assertIn('n_episodes: 8', source)
        self.assertIn('EARLY_STOP_EVALS = 3', source)
        self.assertNotIn("prepare_pick_residual_rl(anchor_exp, 'hard'", source)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['metadata'].get('id','cell'), 'exec')

    def test_notebook_uses_sparse_success_rl_and_freezes_easy(self):
        notebook = make_notebook()
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        self.assertNotIn('drive.mount', source)
        self.assertIn("prepare_success_rl(anchor_exp, 'medium'", source)
        self.assertIn("prepare_success_rl(anchor_exp, 'hard'", source)
        self.assertNotIn("prepare_success_rl(anchor_exp, 'easy'", source)
        self.assertIn("MEDIUM_RL_ITERATIONS = 512", source)
        self.assertIn("HARD_RL_ITERATIONS = 64", source)
        self.assertIn("RL_EVAL_EVERY = 32", source)
        self.assertIn("for stop in evaluation_stops(MEDIUM_RL_ITERATIONS)", source)
        self.assertIn("for stop in evaluation_stops(HARD_RL_ITERATIONS)", source)
        self.assertIn("result['score'] > best['score']", source)
        self.assertIn("'official_best.pt'", source)
        self.assertIn("until_iteration=stop", source)
        self.assertIn("MEDIUM_BEST['source'] == 'success_rl'", source)
        self.assertIn("HARD_BEST['source'] == 'success_rl'", source)
        self.assertIn('Medium BEST-CASE DEMO', source)
        self.assertIn("Path(MEDIUM_BEST['log']).parent/'videos'", source)
        self.assertIn('imitation loss를 사용하지 않습니다', source)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['metadata'].get('id','cell'), 'exec')

    def test_discounted_returns_uses_only_observed_sparse_rewards(self):
        rewards = torch.tensor([[0.,0.],[1.,0.],[0.,2.]])
        result = discounted_returns(rewards, .5)
        torch.testing.assert_close(result, torch.tensor([[.5,.5],[1.,1.],[0.,2.]]))
        torch.testing.assert_close(
            sparse_success_delta(torch.tensor([2,1,0]), torch.tensor([1,1,2])),
            torch.tensor([1.,0.,0.]))
        extreme_kl = bernoulli_kl_from_logits(torch.tensor([1000.,-1000.]),
                                              torch.tensor([-1000.,1000.]))
        self.assertTrue(torch.isfinite(extreme_kl).all())
        self.assertTrue((extreme_kl > 0).all())
        rl_source = (ROOT/'ver2/stages/stage_success_rl.py').read_text(encoding='utf-8')
        self.assertIn("env.unwrapped.evaluate()['success_count']", rl_source)
        self.assertIn('obs, _, _, _, _ = env.step(action)', rl_source)
        self.assertNotIn('action_loss(', rl_source)
        self.assertNotIn('stage_loss(', rl_source)
        self.assertIn("raise FloatingPointError('Non-finite PPO loss", rl_source)
        self.assertIn("resume.get('rl_iteration', resume.get('iteration'))", rl_source)

    def test_package_uses_one_policy_and_three_difficulty_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {name:(ROOT/path).read_text(encoding='utf-8') for name,path in {
                'stage_policy.py':'ver2/stages/stage_policy.py',
                'stage_model.py':'ver2/stages/stage_model.py',
                'stage_schema.py':'ver2/stages/stage_schema.py',
                'act_v2_model.py':'ver2/act_v2_model.py'}.items()}
            for level, dim in DIMS.items():
                cfg = dict(state_dim=dim, history=2, chunk_size=4, width=32,
                           heads=4, layers=1, latent_dim=4)
                model = StageACT(cfg)
                folder = root/level
                folder.mkdir(parents=True)
                torch.save(dict(format='moveboxes-stage-act-v1', model_config=cfg,
                                model=model.state_dict(), step=1), folder/'anchor.pt')
            child = SimpleNamespace(run_dir=root, sources=sources,
                                    cfg={'run_name':'test','team':'team'})
            candidate = package(child)
            manifest = json.loads((candidate/'manifest.json').read_text())
            self.assertEqual(set(manifest['levels']), set(ANCHORS))
            self.assertTrue(all(v['selection']=='anchor' for v in manifest['levels'].values()))
            submission = (candidate/'submission.yaml').read_text()
            self.assertEqual(submission.count('policy: stage_policy:load_policy'), 1)
            for level in ANCHORS:
                self.assertTrue((candidate/'checkpoints'/level/'model.pt').is_file())
            override = root/'medium/rl.pt'
            torch.save(torch.load(root/'medium/anchor.pt', weights_only=True), override)
            selected = package(child, checkpoint_overrides={'medium':override}, folder_name='selected')
            selected_manifest = json.loads((selected/'manifest.json').read_text())
            self.assertEqual(selected_manifest['levels']['medium']['selection'], 'success_rl')
            self.assertEqual(selected_manifest['levels']['easy']['selection'], 'anchor')

    def test_pick_residual_is_zero_initialized_and_group_credit_is_local(self):
        cfg = dict(state_dim=54, history=2, chunk_size=4, width=32,
                   heads=4, layers=1, latent_dim=4, residual_hidden=16,
                   residual_scale=.12)
        base = StageACT(cfg)
        residual = PickResidualStageACT(cfg)
        residual.load_state_dict(base.state_dict(), strict=False)
        obs = torch.randn(3, 2, 54)
        previous = torch.zeros(3, dtype=torch.long)
        stage = torch.tensor([0, 1, 2])
        with torch.no_grad():
            bx, bm, _, _ = base.encode(obs, previous)
            rx, rm, _, _ = residual.encode(obs, previous)
            expected, _ = base.decode(bx, bm, stage)
            actual, _ = residual.decode(rx, rm, stage)
        torch.testing.assert_close(actual, expected)
        advantages = group_relative_advantages(torch.tensor([0., 1., 2., 3., 4., 4., 4., 4.]), 4)
        torch.testing.assert_close(advantages[:4].mean(), torch.tensor(0.))
        torch.testing.assert_close(advantages[4:], torch.zeros(4))
        stages = torch.tensor([[1],[0],[0],[0],[1],[0],[0]])
        self.assertEqual(trailing_pick_mask(stages, 2).squeeze(1).tolist(),
                         [False, False, True, True, False, True, True])

    def test_group_update_changes_only_the_residual(self):
        torch.manual_seed(7)
        cfg = dict(state_dim=54, history=2, chunk_size=4, width=32,
                   heads=4, layers=1, latent_dim=4, residual_hidden=16,
                   residual_scale=.12)
        anchor = StageACT(cfg)
        model = PickResidualStageACT(cfg)
        model.load_state_dict(anchor.state_dict(), strict=False)
        model.requires_grad_(False); model.pick_residual.requires_grad_(True)
        time, envs = 3, 4
        histories = torch.randn(time, envs, 2, 54)
        previous = torch.zeros(time, envs, dtype=torch.long)
        stages = torch.zeros(time, envs, dtype=torch.long)
        raw, old_log = [], []
        with torch.no_grad():
            for step in range(time):
                x, memory, _, _ = model.encode(histories[step], previous[step])
                prediction, _ = model.decode(x, memory, stages[step])
                action = prediction[:, 0, :3]+.01*torch.randn(envs, 3)
                raw.append(action)
                old_log.append(torch.distributions.Normal(prediction[:, 0, :3], .04)
                               .log_prob(action).sum(-1))
        rollout = dict(history=histories, previous_stage=previous, stage=stages,
            raw_xyz=torch.stack(raw), log_prob=torch.stack(old_log),
            reward=torch.zeros(time, envs), score=torch.tensor([0.,1.,2.,3.]),
            credit=torch.ones(time, envs, dtype=torch.bool))
        frozen_before = {name:value.detach().clone() for name,value in model.state_dict().items()
                         if not name.startswith('pick_residual.')}
        optimizer = torch.optim.Adam(model.pick_residual.parameters(), lr=1e-3)
        metrics = group_update(model, anchor, rollout,
            dict(group_size=4, update_epochs=1, minibatch_size=32, xyz_std=.04,
                 clip_ratio=.1, anchor_kl_weight=.2), optimizer, torch.device('cpu'))
        self.assertFalse(metrics['skipped'])
        self.assertTrue(any(value.abs().sum() > 0 for value in model.pick_residual[-1].parameters()))
        for name, before in frozen_before.items():
            torch.testing.assert_close(model.state_dict()[name], before)


if __name__ == '__main__':
    unittest.main()
