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


class AnchorTests(unittest.TestCase):
    def test_notebook_freezes_easy_and_keeps_training_independent(self):
        notebook = make_notebook()
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        self.assertNotIn('drive.mount', source)
        self.assertIn("train(anchor_exp, 'medium')", source)
        self.assertIn("train(anchor_exp, 'hard')", source)
        self.assertNotIn("train(anchor_exp, 'easy')", source)
        self.assertIn("USE_TRAINED = {'medium':False, 'hard':False}", source)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['metadata'].get('id','cell'), 'exec')

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


if __name__ == '__main__':
    unittest.main()
