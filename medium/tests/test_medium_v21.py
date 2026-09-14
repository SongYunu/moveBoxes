import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'medium/tests'),str(ROOT/'medium/code'),str(ROOT/'medium/code/curriculum'),
    str(ROOT/'medium/code/deadline'),str(ROOT/'ver2/stages'),str(ROOT/'ver2'),str(ROOT)]
import numpy as np
import torch
from test_medium_v2 import state, arrays
from curriculum_collect import collect_episode, collect
from curriculum_data import load_data, StageWindows
from curriculum_train import train
from curriculum_eval import Observer, main as evaluate
from medium_v21 import MediumV21, source_bundle, rank
from build_medium_v21_notebook import CONFIG, make_notebook
from stage_model import StageACT, stage_loss
from marso_experiment import digest


def dataset(folder):
    folder.mkdir(parents=True,exist_ok=True);entries=[]
    for i,(correct,duration) in enumerate(((3,10),(3,19),(4,14),(4,18))):
        a=arrays();a['obs'][:,0]=i
        file=folder/f'episode_{i}.npz';np.savez_compressed(file,**a)
        entries.append(dict(file=file.name,sha256=digest(file),accepted=True,correct=correct,
            fully_successful=correct==4,minimum_correct=3,evaluation_actions=199,
            completion_steps={str(correct):duration},failure=None))
    path=folder/'manifest.json'
    path.write_text(json.dumps(dict(version='curriculum-v21',minimum_correct=3,complete=True,episodes=entries)))
    return path


class CurriculumTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_partial_acceptance_never_becomes_four_box_success(self):
        class Env:
            def __init__(self,final):
                self.final=final;self.num_parcels=4;self.parcels=range(4);self.unwrapped=self
                self.agent=SimpleNamespace(is_grasping=lambda _:torch.tensor([False]))
            def obs(self):
                s=state();s[0]=self.t;return torch.from_numpy(s[None])
            def reset(self,seed):self.t=0;return self.obs(),{}
            def step(self,a):
                self.t+=1;return self.obs(),0,torch.tensor([False]),torch.tensor([False]),{}
            def evaluate(self):return dict(success_count=torch.tensor([self.final if self.t>=80 else 0]))
        class Teacher:
            phase='above'
            def __init__(self,*a):pass
            def action(self,*a):return np.array([0,0,0,1],np.float32),False
        with patch('curriculum_collect.DeadlineTeacher',Teacher):
            for final in (2,3,4):
                env=Env(final);_,r=collect_episode(env,1,1.25)
                self.assertEqual(env.t,199)
                self.assertEqual(r['accepted'],final>=3)
                self.assertEqual(r['fully_successful'],final==4)
                self.assertEqual(r['completion_steps'][str(final)],80)

    def test_efficiency_bonus_only_for_full_successes_and_training_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=dataset(Path(tmp));data=load_data(path)
            windows=StageWindows(data,[0,1,2,3],4,16,efficiency_bonus=.5)
            self.assertEqual(windows.efficiency,{0:1.,1:1.,2:1.5,3:1.})
            partial=StageWindows(data,[0,1],4,16)
            self.assertEqual(partial.efficiency,{0:1.,1:1.})
            train_only=StageWindows(data,[0,2],4,16)
            self.assertEqual(train_only.efficiency,{0:1.,2:1.})
            valid=StageWindows(data,[2,3],4,16,training=False)
            self.assertEqual(valid.efficiency,{2:1.,3:1.})
            batch=windows.batch(64,torch.Generator().manual_seed(1),'cpu')
            self.assertIn(1.5,batch['efficiency_weight'].tolist())
            self.assertTrue(torch.all(batch['efficiency_weight'][batch['obs'][:,-1,0]<2]==1))
            m=json.loads(path.read_text());m['episodes'][0]['fully_successful']=True;path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):load_data(path)

    def test_weighted_action_loss_changes_gradient_not_success_labels(self):
        prediction=torch.zeros(2,1,4,requires_grad=True)
        target=torch.ones(2,1,4)
        outputs=(prediction,torch.tensor(0.),torch.zeros(2,4),torch.zeros(2,3))
        cfg=dict(kl_weight=0.,stage_loss_weight=0.,gate_loss_weight=0.)
        loss,_=stage_loss(outputs,target,torch.tensor([[1.5],[1.]]),torch.zeros(2,dtype=torch.long),torch.zeros(2,dtype=torch.long),cfg)
        loss.backward()
        self.assertAlmostEqual(float(prediction.grad[0,0,0]/prediction.grad[1,0,0]),1.5,places=6)

    def test_collection_can_extend_budget_and_does_not_repeat_pilots(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(gains=[1.25],pilot_episodes=1,pilot_seed_start=100,seed_start=200,minimum_correct=3,
                max_attempts=4,episodes=4,noise_std=0.,noise_probability=0.,seconds_per_call=999)
            job=dict(operation='collect',folder=tmp,collection_config=cfg,source_sha256='test')
            utils=SimpleNamespace(compose_cfg=lambda x:SimpleNamespace(randomization={}),
                _gym_make=lambda *a:SimpleNamespace(close=lambda:None))
            seen=[]
            def episode(env,seed,gain,*args,**kwargs):
                seen.append(seed);correct=2 if seed in (100,200) else 3
                return arrays(),dict(seed=seed,gain=gain,accepted=correct>=3,correct=correct,num_parcels=4,steps=20,
                    fully_successful=False,minimum_correct=3,evaluation_actions=199,completion_steps={str(correct):18},
                    first_complete_step=None,failure=None)
            with patch.dict(sys.modules,{'warehouse_sort.utils':utils}),patch('curriculum_collect.collect_episode',side_effect=episode),patch('curriculum_collect.sync_from_env'):
                collect(job)
                m=json.loads((Path(tmp)/'manifest.json').read_text())
                self.assertEqual(len(m['episodes']),3);self.assertFalse(m['complete'])
                cfg['max_attempts']=8;cfg['pilot_episodes']=2;collect(job)
                self.assertEqual(seen,[100,200,201,202,203,101,204])
                collect(job);self.assertEqual(len(seen),7)
            self.assertEqual(len(load_data(Path(tmp)/'manifest.json')),4)

    def test_insufficient_data_is_a_saved_wait_not_a_training_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=MediumV21(dict(CONFIG,profile='smoke',output_root=tmp,repo_dir=tmp,project_dir=str(ROOT)),source_bundle())
            exp.connect()
            with patch.object(exp,'collect_deadline') as collect_more,patch.object(exp,'training_job') as training:
                self.assertIsNone(exp.run_blocks())
                collect_more.assert_called_once_with('collect');training.assert_not_called()

    def test_model_selection_uses_time_only_after_success_and_full_success(self):
        base=dict(score=.75,all_sorted_rate=0.,first_grasp_rate=1.,mean_successful_actions=None,mean_completion_actions=180)
        faster_partial=dict(base,mean_completion_actions=100)
        self.assertEqual(rank(base),rank(faster_partial))
        full=dict(base,score=1.,all_sorted_rate=1.,mean_successful_actions=170)
        fast_full=dict(full,mean_successful_actions=150)
        self.assertGreater(rank(fast_full),rank(full))
        self.assertGreater(rank(full),rank(dict(faster_partial,mean_successful_actions=1)))

    def test_observer_records_actual_action_count_and_resets(self):
        policy=SimpleNamespace(reset=lambda:None)
        observer=Observer(policy);count=[0]
        observer.env=SimpleNamespace(unwrapped=SimpleNamespace(evaluate=lambda:dict(success_count=torch.tensor(count))))
        observer.capture_progress(0);count[0]=3;observer.capture_progress(150)
        count[0]=4;observer.capture_progress(180);observer.capture_progress(199)
        self.assertEqual(observer.completion_steps,{'1':150,'2':150,'3':150,'4':180})
        observer.reset();self.assertEqual(observer.completion_steps,{})
        observer.env=None;observer.capture_progress(5)

    def test_evaluator_averages_time_only_for_full_success_and_video_uses_own_env(self):
        class Env:
            single_action_space=SimpleNamespace(shape=(4,))
            def __init__(self):self.unwrapped=self;self.closed=False
            def reset(self,seed):
                self.t=0;self.final=3 if seed==54000 else 4
                return torch.from_numpy(state()[None]),{}
            def evaluate(self):
                self.assert_open()
                return dict(success_count=torch.tensor([4 if self.t>=180 and self.final==4 else 3 if self.t>=100 else 0]))
            def assert_open(self):
                if self.closed:raise RuntimeError('Closed evaluation environment')
            def close(self):self.closed=True
        class Policy:
            def reset(self):pass
            def act(self,obs,deterministic=True):
                self.last_decision={k:torch.tensor([0]) for k in ('previous','stage','proposed_stage','gate','accepted','gate_confidence','stage_confidence')}
                return torch.tensor([[0.,0.,0.,1.]])
        def rollout(env,agent,device,count,seeds,max_steps):
            self.assertEqual(max_steps,200);obs,_=env.reset(seeds[0])
            for i in range(199):agent.act(obs);env.t+=1
            correct=int(env.evaluate()['success_count'][0])
            return dict(sort_accuracy=correct/4,mean_sorted=correct,all_placed_rate=int(correct==4),mean_steps=100,mis_sort_rate=0)
        def video(cfg,mode,randomization,agent,*args,**kwargs):
            self.assertIsNone(agent.env)
            agent.act(torch.from_numpy(state()[None]))
        utils=SimpleNamespace(compose_cfg=lambda x:SimpleNamespace(randomization={}),make_env=lambda *a,**k:(Env(),None),
            rollout_metrics=rollout,record_eval_video=video)
        with tempfile.TemporaryDirectory() as tmp:
            job=dict(level='medium',checkpoint='fake',policy_config={},max_steps=200,seeds=[54000,54001],
                fingerprint='fixture',output=str(Path(tmp)/'metrics.json'))
            with patch.dict(sys.modules,{'warehouse_sort.utils':utils}),patch('stage_policy.load_stage',return_value=Policy()),patch('curriculum_eval.sync_from_env'):
                evaluate(job)
                metrics=json.loads(Path(job['output']).read_text())
                self.assertEqual(metrics['sort_accuracy'],.875)
                self.assertEqual(metrics['mean_successful_actions'],180)
                self.assertEqual(metrics['mean_completion_actions'],140)
                self.assertIsNone(metrics['episodes'][0]['successful_actions'])
                evaluate(dict(job,video_only=True,seeds=[54000],fingerprint='fixture_video',output=str(Path(tmp)/'video.json')))

    def test_bootstrap_runs_fresh_and_after_old_commit(self):
        nb=make_notebook();self.assertEqual(len(nb['cells']),12)
        for c in nb['cells']:compile(''.join(c['source']),'cell','exec')
        for name,source in source_bundle().items():compile(source,name,'exec')
        for stale in (None,'b'*40):
            scope={};exec(''.join(nb['cells'][0]['source']),scope)
            scope['CFG'].update(project_dir=str(ROOT),profile='smoke')
            if stale:scope['CFG']['project_commit']=stale
            def git(command,**kwargs):return 'https://github.com/SongYunu/moveBoxes.git\n' if 'remote' in command else 'a'*40+'\n'
            previous=list(sys.path)
            try:
                with patch('subprocess.run'),patch('subprocess.check_output',side_effect=git),patch('importlib.reload',side_effect=lambda m:m),contextlib.redirect_stdout(io.StringIO()):
                    exec(''.join(nb['cells'][1]['source']),scope)
                c=scope['experiment'].cfg
                self.assertEqual(c['project_commit'],'a'*40)
                self.assertEqual((c['minimum_correct'],c['efficiency_bonus'],c['total_iters']['medium']),(3,.5,12000))
                self.assertEqual(c['deadline_episodes'],32)
            finally:sys.path[:]=previous

    def test_cpu_training_resume_eval_and_partial_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=dict(CONFIG,profile='smoke',output_root=str(root/'runs'),repo_dir=str(root/'repo'),project_dir=str(ROOT),
                block_iters=4,max_blocks=2,batch_size=4,history=4,width=32,heads=4,layers=1,latent_dim=8,
                validation_batches=1,amp=False,test_record_video=False,record_eval_video=False)
            exp=MediumV21(cfg,source_bundle());exp.connect();exp.repo.mkdir()
            folder=exp.run_dir/'medium';dataset(folder/'collection');(folder/'inputs_ready.json').write_text('{}')
            initial=folder/'checkpoints/initial_model.pt';initial.parent.mkdir()
            model=StageACT(exp.model_config('medium'))
            torch.save(dict(format='moveboxes-stage-act-v1',model_config=model.cfg,model=model.state_dict(),step=4000),initial)
            original=exp.training_job
            def job_cpu():return dict(original(),device='cpu')
            fail=[True]
            def run(command,cwd=None,log=None):
                job=json.loads(Path(command[-1]).read_text())
                if command[1]=='stage_train.py':
                    with patch('curriculum_train.sync_from_env'):train(job)
                    return
                if 'dev_b02' in job['output'] and fail[0]:
                    fail[0]=False;raise RuntimeError('interrupted')
                score=.75 if 'block_01' in job['checkpoint'] else .25
                row=dict(sort_accuracy=score,mean_sorted=4*score,all_placed_rate=0,mean_steps=100,mis_sort_rate=0,
                    mean_completion_actions=170,mean_successful_actions=None,next_pick=dict(stable_grasp_cycles=1,gripper_sign_reversals=2))
                Path(job['output']).write_text(json.dumps(dict(row,episodes=[dict(row,seed=s) for s in job['seeds']],
                    n_episodes=len(job['seeds']),complete=True,protocol=job)))
            with patch.object(exp,'training_job',side_effect=job_cpu),patch.object(exp,'run',side_effect=run):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):exp.run_blocks()
                exp.run_blocks();self.assertEqual(exp._test_candidate('medium')[0].name,'block_01.pt')
                self.assertEqual(exp.final_evaluation()['sort_accuracy'],.75)
            with zipfile.ZipFile(exp.package()) as archive:
                self.assertFalse(any(Path(n).name.startswith(('curriculum_','deadline_')) for n in archive.namelist()))
                self.assertIn('stage_policy.py',archive.namelist())
