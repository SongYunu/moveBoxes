import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ver2/stages'),str(ROOT/'ver2'),str(ROOT)]
import h5py
import numpy as np
import torch
from stage_schema import PICK, CARRY, PLACE, DONE, HOLD, COMPLETE, RECOVER
from stage_labels import StageLabels
from stage_policy import StagePolicy
from stage_model import StageACT
from stage_data import load_data, split_data, StageWindows
from stage_teacher import CorrectiveTeacher
from stage_collect import collect_episode
from stage_train import train, training_forward
from stage_runtime_check import run_checks
from stage_experiment import StageExperiment, source_bundle
from build_stage_notebook import CONFIG, make_notebook, make_repair_notebook


def state():
    obs = np.zeros(54, dtype=np.float32)
    obs[18:21] = [.2,0,.22]
    obs[26:29] = [.2,0,.03]
    obs[33:36] = [.2,.12,.03]
    obs[40:44] = [1,0,0,1]
    obs[44:50] = [0,-.36,0,0,.36,0]
    return obs


class StageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_shapes_backward_checkpoint_reset_all_levels(self):
        run_checks()

    def test_training_action_path_matches_execution_and_does_not_read_expert_future(self):
        model=StageACT(dict(state_dim=54,history=4,chunk_size=16,width=32,heads=4,layers=1,latent_dim=8))
        batch=dict(obs=torch.zeros(2,4,54),previous_stage=torch.zeros(2,dtype=torch.long),
                   stage=torch.zeros(2,dtype=torch.long),actions=torch.randn(2,16,4),mask=torch.ones(2,16))
        first=training_forward(model,batch,dict(action_training_mode='prior'))[0]
        batch['actions'] *= 100
        second=training_forward(model,batch,dict(action_training_mode='prior'))[0]
        x,memory,_,_=model.encode(batch['obs'],batch['previous_stage'])
        actual,_=model.decode(x,memory,batch['stage'])
        torch.testing.assert_close(first,second,atol=0,rtol=0)
        torch.testing.assert_close(first,actual,atol=0,rtol=0)
        first.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.posterior.parameters()))
        self.assertIsNotNone(model.obs_proj.weight.grad)

    def test_gate_holds_until_confident_and_clears_old_chunk_on_recovery(self):
        class Controlled(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.cfg = dict(history=2,chunk_size=4)
            def encode(self, obs, previous):
                phase = torch.full((len(obs),4), -20.)
                gate = torch.full((len(obs),3), -20.)
                for i,row in enumerate(obs[:,-1]):
                    phase[i,int(row[0])] = 20
                    if row[2] == 1:
                        gate[i] = 0  # An uncertain completion cannot advance.
                    else:
                        gate[i,int(row[1])] = 20
                return obs,obs,phase,gate
            def decode(self, obs, memory, phase):
                prediction = obs.new_zeros(len(obs),4,4)
                prediction[...,:3] = phase[:,None,None]*.4
                prediction[...,3] = 1
                return prediction,None
        p = StagePolicy(Controlled(), ensemble_window=4)
        obs = torch.zeros(2,54)
        obs[:,0], obs[:,1], obs[:,2] = CARRY,COMPLETE,1
        self.assertEqual(float(p.act(obs).sum()),2.)  # open grippers, no motion
        self.assertTrue(torch.all(p.stage == PICK))
        obs[0,2] = 0
        action = p.act(obs)
        self.assertAlmostEqual(float(action[0,0]),.4,places=6)
        self.assertEqual(float(action[1,0]),0.)
        obs[0,0],obs[0,1] = PICK,RECOVER
        self.assertEqual(float(p.act(obs)[0,0]),0.)
        self.assertEqual(int(p.stage[0]),PICK)
        p.reset()
        self.assertIsNone(p.last_decision)

    def test_label_completion_needs_grasp_lift_and_verified_release(self):
        labeler = StageLabels(2)
        s = state()
        for _ in range(5):
            self.assertEqual(labeler.observe(s)['stage'], PICK)
        s[25],s[20],s[28] = 1,.26,.19
        labeler.observe(s)
        self.assertEqual(labeler.observe(s)['stage'],CARRY)
        s[18:21],s[26:29] = [0,-.36,.26],[0,-.36,.19]
        self.assertEqual(labeler.observe(s)['stage'],PLACE)
        s[25],s[28] = 0,.03
        self.assertEqual(labeler.observe(s)['stage'],PLACE)
        result = labeler.observe(s)
        self.assertEqual((result['stage'],result['target'],result['gate']),(PICK,1,COMPLETE))

    def test_teacher_timeout_retries_same_parcel_and_drop_does_not_advance(self):
        teacher = CorrectiveTeacher(2)
        teacher.phase,teacher.ticks = 'above',86
        _,recovering = teacher.action(state(),[False,False])
        self.assertTrue(recovering)
        self.assertEqual(teacher.target,0)
        self.assertEqual(teacher.phase,'retreat')
        teacher.phase,teacher.ticks = 'carry',0
        _,recovering = teacher.action(state(),[False,False])
        self.assertTrue(recovering)
        self.assertEqual(teacher.target,0)

    def test_collection_keeps_clean_labels_and_pre_action_alignment(self):
        class Env:
            def __init__(self):
                self.unwrapped,self.num_parcels,self.parcels = self,2,[0,1]
                self.agent=SimpleNamespace(is_grasping=lambda p:torch.tensor([False]))
            def observation(self):
                s = state();s[0]=self.t
                return torch.from_numpy(s[None])
            def reset(self,seed):
                self.t=0
                return self.observation(),{}
            def step(self, action):
                self.t += 1
                return self.observation(),torch.zeros(1),torch.tensor([False]),torch.tensor([False]),{}
            def evaluate(self):
                return dict(success_count=torch.tensor([0]))
        arrays,report = collect_episode(Env(),123,dict(max_steps=4,noise_probability=1,action_noise_std=.2,drop_probability=0))
        np.testing.assert_array_equal(arrays['obs'][:,0],np.arange(5))
        self.assertEqual(arrays['actions'].shape,(4,4))
        self.assertTrue(np.any(arrays['actions'] != arrays['executed_actions']))
        self.assertFalse(report['accepted'])

    def make_data(self,path):
        with h5py.File(path,'w') as f:
            for i in range(10):
                group = f.create_group(f'traj_{i}')
                obs = np.repeat(state()[None],13,axis=0)
                obs[:,0]=i
                actions = np.zeros((12,4),dtype=np.float32)
                actions[:,0]=2;actions[:,3]=1
                group.create_dataset('obs',data=obs)
                group.create_dataset('actions',data=actions)

    def test_windows_do_not_cross_phase_target_or_disturbance_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.h5';self.make_data(path)
            data=load_data(path)
            train_ids,val_ids=split_data(data,42)
            self.assertFalse(set(train_ids)&set(val_ids))
            data[0]['stage'][3:]=CARRY
            data[0]['target'][6:]=1
            data[0]['perturbed'][9]=True
            windows=StageWindows(data,[0],4,16,training=False)
            for t,expected in ((0,3),(3,3),(6,4),(9,1)):
                windows.indices=[(0,t)]
                batch=windows.batch(1,torch.Generator().manual_seed(0),'cpu')
                self.assertEqual(int(batch['mask'].sum()),expected)
                self.assertEqual(float(batch['actions'][0,0,0]),1.)

    def test_resume_restores_optimizer_and_rng_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'data.h5';self.make_data(path)
            cfg=dict(seed=42,batch_size=4,lr=.001,total_iters=8,save_freq=4,warmup_steps=2,
                kl_weight=.001,stage_loss_weight=.3,gate_loss_weight=.3,position_noise=.001,
                validation_batches=1,amp=False,console_interval_seconds=999)
            arch=dict(state_dim=54,history=4,chunk_size=16,width=32,heads=4,layers=1,latent_dim=8)
            job=dict(data=str(path),folder=str(root/'full'),model_config=arch,train_config=cfg,
                source_sha256='test',policy_config=dict(model_config=arch),device='cpu')
            with patch('stage_train.sync_from_env'):
                train(job)
            full=torch.load(root/'full/checkpoints/latest.pt',weights_only=True)
            job['folder']=str(root/'resume')
            with patch('stage_train.sync_from_env',side_effect=RuntimeError('disconnect')):
                with self.assertRaisesRegex(RuntimeError,'disconnect'):
                    train(job)
            with patch('stage_train.sync_from_env'):
                train(job)
            resumed=torch.load(root/'resume/checkpoints/latest.pt',weights_only=True)
            for key in full['model']:
                torch.testing.assert_close(full['model'][key],resumed['model'][key],atol=0,rtol=0)

    def test_notebook_all_levels_and_package_contains_only_learned_controller(self):
        nb=make_notebook()
        self.assertEqual(len(nb['cells']),22)
        for i,c in enumerate(nb['cells']):
            self.assertEqual(c['cell_type'],'code')
            compile(''.join(c['source']),str(i),'exec')
        sources=source_bundle()
        for name,code in sources.items():
            compile(code,name,'exec')
        repair=make_repair_notebook()
        self.assertEqual(len(repair['cells']),12)
        for c in repair['cells']:
            compile(''.join(c['source']),'repair','exec')
        scope={}
        exec(''.join(repair['cells'][0]['source']),scope)
        self.assertEqual(scope['CFG']['run_name'],'moveboxes_stage_act_v1')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=dict(CONFIG,profile='smoke',output_root=str(root/'runs'),repo_dir=str(root/'repo'),
                     project_dir=str(ROOT),test_record_video=False,record_eval_video=False)
            exp=StageExperiment(cfg,sources);exp.connect()
            exp.repo.mkdir();exp._stage_helpers()
            self.assertEqual((exp.repo/'colab_eval_modular.py').read_text(encoding='utf-8'),sources['stage_eval.py'])
            def fake_eval(command,cwd=None,log=None):
                job=json.loads(Path(command[-1]).read_text())
                self.assertEqual(job['max_steps'],200)
                row=dict(sort_accuracy=.5,mean_sorted=1,all_placed_rate=0,mean_steps=100,mis_sort_rate=0)
                episodes=[dict(row,seed=s) for s in job['seeds']]
                Path(job['output']).write_text(json.dumps(dict(row,complete=True,n_episodes=len(episodes),episodes=episodes,protocol=job)))
            for level in ('easy','medium','hard'):
                folder=exp.run_dir/level
                ckdir=folder/'checkpoints';ckdir.mkdir(parents=True)
                model=StageACT(exp.model_config(level))
                torch.save(dict(format='moveboxes-stage-act-v1',model_config=model.cfg,model=model.state_dict()),ckdir/'best_val.pt')
                (folder/'stage_train_job.json').write_text(json.dumps(dict(policy_config=exp.policy_config(level))))
                with patch.object(exp,'run',side_effect=fake_eval):
                    exp.test(level);exp.evaluate(level)
            with zipfile.ZipFile(exp.package()) as archive:
                self.assertNotIn('stage_teacher.py',archive.namelist())
                self.assertNotIn('stage_collect.py',archive.namelist())
                self.assertIn('stage_model.py',archive.namelist())
                self.assertIn('from stage_model import StageACT',archive.read('colab_policy.py').decode())
                submission=json.loads(archive.read('submission.yaml'))
                self.assertEqual(set(submission['state']['levels']),{'easy','medium','hard'})

    def test_zero_quick_test_stops_expensive_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=dict(CONFIG,profile='smoke',output_root=directory,project_dir=str(ROOT))
            exp=StageExperiment(cfg,source_bundle());exp.connect()
            ck=exp.run_dir/'easy/checkpoints/latest.pt';ck.parent.mkdir(parents=True);ck.write_bytes(b'fixture')
            with patch.object(exp,'_saved_final',return_value=None), \
                 patch.object(exp,'_test_candidate',return_value=(ck,exp.policy_config('easy'))), \
                 patch.object(exp,'test',return_value=dict(sort_accuracy=0)), \
                 patch.object(exp,'_trial') as trial:
                self.assertIsNone(exp.evaluate('easy'))
                trial.assert_not_called()
            self.assertFalse(json.loads((exp.run_dir/'easy/evaluation_gate.json').read_text())['passed'])

    def test_repair_reuses_data_and_weights_in_an_isolated_run(self):
        from marso_experiment import digest
        with tempfile.TemporaryDirectory() as directory:
            cfg=dict(CONFIG,run_name='original',profile='smoke',output_root=str(Path(directory)/'runs'),
                     repo_dir=str(Path(directory)/'repo'),project_dir=str(ROOT))
            exp=StageExperiment(cfg,source_bundle());exp.connect();exp.repo.mkdir()
            source=exp.run_dir/'easy'
            (source/'checkpoints').mkdir(parents=True)
            (source/'checkpoints/latest.pt').write_bytes(b'original-checkpoint')
            (source/'stage_train_job.json').write_text(json.dumps(dict(model_config=exp.model_config('easy'))))
            collection=source/'collection';collection.mkdir()
            episode=collection/'episode.npz';episode.write_bytes(b'original-episode')
            manifest=dict(complete=True,episodes=[dict(file=episode.name,sha256=digest(episode))])
            (collection/'manifest.json').write_text(json.dumps(manifest))
            with patch.object(StageExperiment,'train') as training, \
                 patch.object(StageExperiment,'test'),patch.object(StageExperiment,'diagnose'):
                fixed=exp.repair('easy')
                training.assert_called_once_with('easy')
            self.assertNotEqual(fixed.run_dir,exp.run_dir)
            self.assertEqual(fixed.cfg['action_training_mode'],'prior')
            self.assertEqual((fixed.run_dir/'easy/initial_model.pt').read_bytes(),b'original-checkpoint')
            self.assertEqual((fixed.run_dir/'easy/collection/episode.npz').read_bytes(),b'original-episode')
            self.assertFalse((source/'repair_origin.json').exists())


if __name__ == '__main__':
    unittest.main()
