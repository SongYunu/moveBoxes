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
from stage_anchor_continue import ANCHORS, DIMS, package
from stage_model import StageACT
from stage_success_rl import bernoulli_kl_from_logits, discounted_returns, sparse_success_delta


class AnchorTests(unittest.TestCase):
    def test_notebook_uses_sparse_success_rl_and_freezes_easy(self):
        notebook = make_notebook()
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        self.assertNotIn('drive.mount', source)
        self.assertIn("prepare_success_rl(anchor_exp, 'medium'", source)
        self.assertIn("prepare_success_rl(anchor_exp, 'hard'", source)
        self.assertNotIn("prepare_success_rl(anchor_exp, 'easy'", source)
        self.assertIn("for stop in (8, 16, 24)", source)
        self.assertIn("result['score'] > best['score']", source)
        self.assertIn("'official_best.pt'", source)
        self.assertIn("until_iteration=stop", source)
        self.assertIn("MEDIUM_BEST['source'] == 'success_rl'", source)
        self.assertIn("HARD_BEST['source'] == 'success_rl'", source)
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


if __name__ == '__main__':
    unittest.main()
