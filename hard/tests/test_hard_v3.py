import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/'ver2',ROOT/'ver2/stages',ROOT/'hard/code'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from stage_model import StageACT
from hard_transfer import feature_pairs,transplant
from build_hard_v3_notebook import make_notebook
from build_hard_v31_notebook import make_notebook as make_v31_notebook


def arch(dim):
    return dict(state_dim=dim,history=4,chunk_size=4,width=32,heads=4,layers=1,latent_dim=4)


class HardV3Test(unittest.TestCase):
    def test_semantic_transplant_preserves_donor_output(self):
        torch.manual_seed(7);donor=StageACT(arch(54)).eval()
        donor.obs_mean.copy_(torch.randn_like(donor.obs_mean));donor.obs_std.copy_(torch.rand_like(donor.obs_std)+.5)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'easy.pt';output=Path(tmp)/'hard.pt'
            torch.save(dict(format='moveboxes-stage-act-v1',model_config=arch(54),model=donor.state_dict(),step=123),source)
            transplant(source,arch(90),torch.zeros(126),torch.ones(126),output)
            saved=torch.load(output,map_location='cpu',weights_only=True);hard=StageACT(arch(90)).eval();hard.load_state_dict(saved['model'])
            easy_obs=torch.randn(2,4,54);hard_obs=torch.zeros(2,4,90)
            raw_pairs=[p for p in feature_pairs(54) if p[0]<54 and p[1]<90]
            for src,dst in raw_pairs:hard_obs[...,dst]=easy_obs[...,src]
            previous=torch.tensor([0,2]);stage=torch.tensor([0,2])
            with torch.no_grad():a=donor(easy_obs,previous,stage)[0];b=hard(hard_obs,previous,stage)[0]
            self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=2e-6))
            self.assertEqual(saved['transfer']['source_step'],123)

    def test_notebook_is_hard_only_and_compiles(self):
        notebook=make_notebook()
        for i,cell in enumerate(notebook['cells']):compile(''.join(cell['source']),str(i),'exec')
        text=''.join(''.join(c['source']) for c in notebook['cells'])
        self.assertIn('HardV3',text);self.assertIn('experiment.test("hard")',text)
        self.assertNotIn('experiment.test("easy")',text)

    def test_v31_uses_only_medium_and_tests_before_training(self):
        notebook=make_v31_notebook()
        for i,cell in enumerate(notebook['cells']):compile(''.join(cell['source']),str(i),'exec')
        text=''.join(''.join(c['source']) for c in notebook['cells'])
        self.assertIn("'transfer_donors': {'medium': 'moveboxes_medium_zfocus_v22'}",text)
        self.assertIn('HardV31',text)
        self.assertLess(text.index('experiment.test_transfer()'),text.index('experiment.run_blocks()'))


if __name__=='__main__':unittest.main()
