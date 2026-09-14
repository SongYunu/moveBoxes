import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_unified import ROOT,state
import torch
from stage_model import StageACT
from stage_policy import load_stage as native_load
from probe_medium_policy import load_stage,medium_observation
from probe_medium import MediumProbe,source_bundle
from build_medium_probe_notebook import make_notebook


class MediumProbeTests(unittest.TestCase):
    def test_unchanged_weights_and_medium_actions(self):
        cfg=dict(state_dim=72,history=4,chunk_size=4,width=32,heads=4,layers=1,latent_dim=4)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';model=StageACT(cfg)
            torch.save(dict(format='moveboxes-stage-act-v1',model_config=cfg,model=model.state_dict()),path)
            before=path.read_bytes();obs=torch.tensor(state(72,1));space=type('A',(),{'shape':(4,)})()
            left=native_load(path,obs,space,'cpu');right=load_stage(path,obs,space,'cpu')
            for _ in range(6):self.assertTrue(torch.equal(left.act(obs),right.act(obs)))
            right.reset();self.assertEqual(right.act(torch.tensor(state(54,1))).shape,(1,4))
            self.assertEqual(before,path.read_bytes())
            easy=torch.tensor(state(54,1));padded=medium_observation(easy)
            self.assertTrue(torch.equal(padded[...,62:],easy[...,44:]))
            self.assertTrue(torch.equal(padded[...,54:58],easy[...,40:44]))
            self.assertEqual(padded[...,40:54].abs().sum(),0)
            with self.assertRaises(ValueError):medium_observation(torch.tensor(state(90,1)))

    def test_notebook_only_evaluates_and_bootstraps(self):
        nb=make_notebook();scope={};exec(''.join(nb['cells'][0]['source']),scope)
        scope['CFG']['project_dir']=str(ROOT)
        def git(cmd,**kw):return 'https://github.com/SongYunu/moveBoxes.git\n' if 'remote' in cmd else 'a'*40+'\n'
        with patch('subprocess.run'),patch('subprocess.check_output',side_effect=git),patch('importlib.reload',side_effect=lambda m:m),contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(nb['cells'][1]['source']),scope)
        self.assertIsInstance(scope['experiment'],MediumProbe)
        for cell in nb['cells']:
            code=''.join(cell['source']);compile(code,'cell','exec')
            self.assertNotIn('experiment.train',code);self.assertNotIn('experiment.calibrate',code)
        for name,code in source_bundle().items():compile(code,name,'exec')


if __name__=='__main__':unittest.main()
