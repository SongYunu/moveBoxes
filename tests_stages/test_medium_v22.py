import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ver2/zfocus'),str(ROOT/'ver2/curriculum'),str(ROOT/'ver2/deadline'),
    str(ROOT/'ver2/stages'),str(ROOT/'ver2'),str(ROOT),str(ROOT/'tests_stages')]
import torch
from zfocus_model import StageACT, stage_loss
from build_medium_v22_notebook import CONFIG, make_notebook
from medium_v22 import MediumV22, source_bundle
from test_medium_v21 import dataset
import curriculum_train


class HeightTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_contact_z_and_gripper_gradients_and_padding(self):
        prediction=torch.zeros(3,2,4,requires_grad=True)
        actions=torch.zeros_like(prediction);actions[:,:,:3]=.05
        actions[0,:,3]=1;actions[1,:,3]=-1;actions[2,:,3]=-1
        cfg=dict(pick_z_weight=3.,contact_z_weight=6.,contact_gripper_weight=3.,
            kl_weight=0.,stage_loss_weight=0.,gate_loss_weight=0.)
        loss,_=stage_loss((prediction,torch.tensor(0.),torch.zeros(3,4),torch.zeros(3,3)),
            actions,torch.tensor([[1.,0.]]*3),torch.tensor([0,0,1]),torch.zeros(3,dtype=torch.long),cfg)
        loss.backward();g=prediction.grad
        self.assertAlmostEqual(float(g[0,0,2]/g[0,0,0]),3.)
        self.assertAlmostEqual(float(g[1,0,2]/g[1,0,0]),6.)
        self.assertAlmostEqual(float(g[2,0,2]/g[2,0,0]),1.)
        self.assertAlmostEqual(float(g[1,0,3]/g[2,0,3]),3.)
        self.assertEqual(float(g[:,1].abs().sum()),0.)

    def test_cpu_resume_preserves_encoder_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=dataset(root/'collection')
            arch=dict(state_dim=72,history=4,chunk_size=16,width=32,heads=4,layers=1,latent_dim=8)
            initial=StageACT(arch);initial_path=root/'initial.pt'
            torch.save(dict(format='moveboxes-stage-act-v1',model_config=arch,model=initial.state_dict(),step=4000),initial_path)
            cfg=dict(seed=42,batch_size=4,lr=1e-4,warmup_steps=1,total_iters=4,save_freq=2,
                validation_batches=1,amp=False,position_noise=0.,console_interval_seconds=999,
                kl_weight=0.,stage_loss_weight=.1,gate_loss_weight=.1,efficiency_bonus=.5,
                pick_z_weight=3.,contact_z_weight=6.,contact_gripper_weight=3.)
            job=dict(folder=str(root/'train'),data=str(data),model_config=arch,train_config=cfg,
                policy_config={},source_sha256='test-zfocus',warm_start=str(initial_path),device='cpu',stop_at=2)
            with patch.object(curriculum_train,'StageACT',StageACT),patch.object(curriculum_train,'stage_loss',stage_loss),\
                    patch.object(curriculum_train,'sync_from_env'),contextlib.redirect_stdout(io.StringIO()):
                curriculum_train.train(job);curriculum_train.train(dict(job,stop_at=4))
            saved=torch.load(root/'train/checkpoints/latest.pt',weights_only=True)
            self.assertEqual(saved['step'],4)
            for name,p in initial.named_parameters():
                if not p.requires_grad:self.assertTrue(torch.equal(p,saved['model'][name]),name)
            self.assertFalse(torch.equal(initial.output[1].weight,saved['model']['output.1.weight']))
            # Original deployment class loads the new weights without new runtime logic.
            from stage_model import StageACT as Original
            Original(arch).load_state_dict(saved['model'])

    def test_fresh_bootstrap_uses_v22_defaults_and_all_sources_compile(self):
        nb=make_notebook();self.assertEqual(len(nb['cells']),11)
        for cell in nb['cells']:compile(''.join(cell['source']),'cell','exec')
        for name,source in source_bundle().items():compile(source,name,'exec')
        scope={};exec(''.join(nb['cells'][0]['source']),scope)
        scope['CFG'].update(project_dir=str(ROOT),profile='smoke')
        def git(command,**kwargs):return 'https://github.com/SongYunu/moveBoxes.git\n' if 'remote' in command else 'a'*40+'\n'
        previous=list(sys.path)
        try:
            with patch('subprocess.run'),patch('subprocess.check_output',side_effect=git),\
                    patch('importlib.reload',side_effect=lambda m:m),contextlib.redirect_stdout(io.StringIO()):
                exec(''.join(nb['cells'][1]['source']),scope)
            exp=scope['experiment'];self.assertIsInstance(exp,MediumV22)
            self.assertEqual(exp.cfg['total_iters']['medium'],4000)
            self.assertEqual(exp.cfg['source_run_name'],'moveboxes_medium_curriculum_v21')
            self.assertEqual(exp.cfg['contact_z_weight'],6.)
        finally:sys.path[:]=previous


if __name__=='__main__':unittest.main()
