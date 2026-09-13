import copy
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ver2'),str(ROOT)]
import h5py
import numpy as np
import torch
from act_v2_model import StateACT, state_features, action_loss
from act_v2_policy import ACTPolicy, load_act
from act_v2_data import load_trajectories, split_trajectories, normalization, WindowData
from act_v2_train import train
from act_v2_runtime_check import run_checks
from act_v2_experiment import ActV2Experiment, source_bundle
from build_act_v2_notebook import CONFIG, make_notebook


class Ver2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_all_difficulties_checkpoint_and_reset_contract(self):
        run_checks()

    def test_masked_tail_does_not_train_fake_stop_actions(self):
        pred = torch.zeros(1,4,4,requires_grad=True)
        target = torch.ones_like(pred)
        mask = torch.tensor([[1.,1.,0.,0.]])
        first,_ = action_loss(pred,target,mask,torch.tensor(0.))
        changed=target.clone();changed[:,2:]=999
        second,_ = action_loss(pred,changed,mask,torch.tensor(0.))
        self.assertEqual(float(first.detach()),float(second.detach()))
        first.backward()
        self.assertEqual(float(pred.grad[:,2:].abs().sum()),0.)

    def test_relative_features_follow_color_target_when_bins_swap(self):
        s = torch.zeros(1,54)
        s[:,18:21] = torch.tensor([.1,.2,.3])
        s[:,26:29] = torch.tensor([.2,.3,.4])
        s[:,40:44] = torch.tensor([1.,0.,0.,1.])
        s[:,44:50] = torch.tensor([0.,-.36,0.,0.,.36,0.])
        features = state_features(s)
        torch.testing.assert_close(features[0,54:57],torch.tensor([.1,.1,.1]))
        torch.testing.assert_close(features[0,60:63],torch.tensor([-.2,-.66,-.4]))

    def test_temporal_average_is_aligned_and_zero_actions_are_valid(self):
        class Scripted(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
                self.cfg=dict(history=2,chunk_size=4)
            def forward(self,obs):
                # Prediction for absolute t is identical across all overlapping chunks.
                start=obs[:,-1,0]
                out=obs.new_zeros(len(obs),4,4)
                out[...,0]=start[:,None]+torch.arange(4,device=obs.device)*.1
                out[...,3]=1
                return out,None
        p=ACTPolicy(Scripted(),temporal_decay=.25,ensemble_window=4)
        for t in range(5):
            obs=torch.zeros(1,54);obs[0,0]=t*.1
            self.assertAlmostEqual(float(p.act(obs)[0,0]),t*.1,places=6)
        p.reset()
        self.assertEqual(float(p.act(torch.zeros(1,54))[0,0]),0.)

    def make_data(self,path):
        with h5py.File(path,'w') as f:
            for i in range(10):
                g=f.create_group(f'traj_{i}')
                obs=np.zeros((13,54),dtype=np.float32);obs[:,0]=i;obs[:,18]=np.arange(13)*.01
                actions=np.zeros((12,4),dtype=np.float32);actions[:,0]=3;actions[:,3]=1
                actions[0,3]=.2235  # Hard demos include a few soft gripper commands.
                g.create_dataset('obs',data=obs);g.create_dataset('actions',data=actions)

    def test_split_clipping_mask_and_training_only_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.h5';self.make_data(path)
            data=load_trajectories(path)
            ids,val=split_trajectories(data,42)
            self.assertFalse(set(ids)&set(val))
            self.assertEqual(data[0]['actions'][0,0],1.)
            self.assertGreater(data[0]['clipped_fraction'],0)
            self.assertAlmostEqual(float(data[0]['actions'][0,3]), .2235, places=5)
            mean,std=normalization(data,ids)
            data[val[0]]['obs'][:]=10000
            mean2,std2=normalization(data,ids)
            torch.testing.assert_close(mean,mean2);torch.testing.assert_close(std,std2)
            windows=WindowData(data,ids,4,16)
            obs,actions,mask=windows.batch(4,torch.Generator().manual_seed(0),'cpu',.001)
            self.assertTrue(torch.all(mask.sum(1)<=12))
            self.assertEqual(float(obs[...,25].abs().sum()),0.)

    def test_training_resume_reproduces_uninterrupted_cpu_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'data.h5';self.make_data(path)
            cfg=dict(seed=42,batch_size=4,lr=1e-3,total_iters=8,save_freq=4,warmup_steps=2,
                kl_weight=.001,position_noise=.001,validation_batches=1,amp=False,console_interval_seconds=999)
            arch=dict(state_dim=54,history=4,chunk_size=16,width=32,heads=4,layers=1,latent_dim=8)
            job=dict(data=str(path),folder=str(root/'full'),model_config=arch,train_config=cfg,
                source_sha256='test-source',policy_config=dict(model_config=arch),device='cpu')
            with patch('act_v2_train.sync_from_env'):
                train(job)
            full=torch.load(root/'full/checkpoints/latest.pt',weights_only=True)
            job['folder']=str(root/'resumed')
            with patch('act_v2_train.sync_from_env',side_effect=RuntimeError('disconnect')):
                with self.assertRaisesRegex(RuntimeError,'disconnect'):
                    train(job)
            with patch('act_v2_train.sync_from_env'):
                train(job)
            resumed=torch.load(root/'resumed/checkpoints/latest.pt',weights_only=True)
            for k in full['model']:
                torch.testing.assert_close(full['model'][k],resumed['model'][k],atol=0,rtol=0)
            job['train_config']=dict(cfg,lr=.01)
            with self.assertRaisesRegex(ValueError,'configuration'):
                train(job)

    def test_notebook_is_additive_separate_and_all_sources_compile(self):
        nb=make_notebook()
        self.assertEqual(len(nb['cells']),19)
        scope={}
        exec(''.join(nb['cells'][0]['source']),scope)
        self.assertEqual(scope['CFG']['run_name'],'moveboxes_act_ver2')
        self.assertNotIn('unet_dims',scope['CFG'])
        self.assertTrue((ROOT/'notebooks/moveboxes_colab.ipynb').exists())
        for i,c in enumerate(nb['cells']):
            self.assertEqual(c['cell_type'],'code');compile(''.join(c['source']),str(i),'exec')
        sources=source_bundle()
        for name,code in sources.items():
            compile(code,name,'exec')
        exp=ActV2Experiment(scope['CFG'],sources)
        self.assertFalse(set(exp.test_seeds())&set(exp.seeds(True)))
        self.assertFalse(set(exp.seeds(True))&set(exp.seeds(False)))
        for level in ('easy','medium','hard'):
            self.assertIn(level,exp.cfg['total_iters'])

    def test_colab_flow_stages_selects_resumes_and_packages_ver2(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cfg=dict(CONFIG, profile='smoke',output_root=str(root/'runs'),repo_dir=str(root/'repo'),
                     project_dir=str(ROOT),test_record_video=False,record_eval_video=False)
            exp=ActV2Experiment(cfg,source_bundle())
            exp.connect()
            exp.repo.mkdir()
            exp._stage_helpers()
            self.assertEqual((exp.repo/'colab_eval_modular.py').read_text(encoding='utf-8'),exp.sources['act_v2_eval.py'])
            self.assertTrue((exp.repo/'act_v2_model.py').exists())
            for level in ('easy','medium','hard'):
                ckdir=exp.run_dir/level/'checkpoints';ckdir.mkdir(parents=True)
                model=StateACT(exp.model_config(level))
                torch.save(dict(format='moveboxes-act-ver2',model_config=model.cfg,
                                model=model.state_dict()),ckdir/'best_val.pt')
                (exp.run_dir/level/'act_train_job.json').write_text(json.dumps(dict(policy_config=exp.policy_config(level))))
            calls=[]
            def fake_rollout(command,cwd=None,log=None):
                job=json.loads(Path(command[-1]).read_text())
                self.assertEqual(command[1],'colab_eval_modular.py')
                self.assertEqual(job['max_steps'],200)
                calls.append(job)
                score=.5 if job['policy_config']['ensemble_window']==4 else .25
                row=dict(sort_accuracy=score,mean_sorted=score*2,all_placed_rate=0,mean_steps=100,mis_sort_rate=0)
                episodes=[dict(row,seed=s) for s in job['seeds']]
                Path(job['output']).write_text(json.dumps(dict(row,complete=True,n_episodes=len(episodes),
                                                              episodes=episodes,protocol=job)))
            with patch.object(exp,'run',side_effect=fake_rollout):
                for level in ('easy','medium','hard'):
                    exp.test(level)
                    metrics=exp.evaluate(level)
                    self.assertEqual(metrics['sort_accuracy'],.5)
                count=len(calls)
                for level in ('easy','medium','hard'):
                    exp.evaluate(level)
                self.assertEqual(len(calls),count)
            package=exp.package()
            with zipfile.ZipFile(package) as archive:
                submission=json.loads(archive.read('submission.yaml'))
                self.assertEqual(set(submission['state']['levels']),{'easy','medium','hard'})
                self.assertEqual(submission['state']['policy'],'colab_policy:load_policy')
                self.assertIn('act_v2_model.py',archive.namelist())
                self.assertIn('act_v2_policy.py',archive.namelist())
                self.assertIn('from act_v2_model import StateACT',archive.read('colab_policy.py').decode())


if __name__ == '__main__':
    unittest.main()
