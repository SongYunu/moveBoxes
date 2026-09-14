import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'medium/tests'),str(ROOT/'medium/code'),str(ROOT/'medium/code/deadline'),
    str(ROOT/'ver2/stages'),str(ROOT/'ver2'),str(ROOT)]
import numpy as np
import torch
from deadline_teacher import DeadlineTeacher
from deadline_collect import collect_episode, collect
from deadline_data import load_data, split_data, StageWindows
from deadline_train import train
from stage_model import StageACT
from medium_v2 import MediumV2, source_bundle
from build_medium_v2_notebook import CONFIG, make_notebook
from marso_experiment import digest


def state():
    s=np.zeros(72,np.float32);s[18:21]=[.2,0,.24]
    s[26:54]=np.array([[.2,i*.08,.03,1,0,0,0] for i in range(4)]).flatten()
    s[54:62]=[1,0,0,1,1,0,0,1];s[62:68]=[0,-.36,0,0,.36,0]
    return s


def arrays():
    obs=np.repeat(state()[None],21,axis=0);obs[:,20]=.1
    actions=np.zeros((20,4),np.float32);actions[:,3]=1
    return dict(obs=obs,actions=actions,executed_actions=actions.copy(),
        previous_stage=np.zeros(20,np.int64),stage=np.zeros(20,np.int64),gate=np.zeros(20,np.int64),
        target=np.repeat(np.arange(4),5),perturbed=np.zeros(20,bool))


def dataset(folder):
    folder.mkdir(parents=True,exist_ok=True);entries=[]
    for i in range(4):
        a=arrays();a['obs'][:,0]=i
        path=folder/f'episode_{i}.npz';np.savez_compressed(path,**a)
        entries.append(dict(file=path.name,sha256=digest(path),accepted=True,correct=4,evaluation_actions=199))
    path=folder/'manifest.json'
    path.write_text(json.dumps(dict(version='deadline-v2',complete=True,episodes=entries)))
    return path


class DeadlineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_teacher_aligns_before_close_and_waits_for_observed_grasp(self):
        teacher=DeadlineTeacher(4,1.75);s=state();s[18]+=.025;s[20]=.063
        teacher.phase='descend'
        action,_=teacher.action(s,[False]*4)
        self.assertEqual(teacher.phase,'descend');self.assertEqual(action[3],1)
        self.assertLess(action[0],0)
        s[18]=.2
        teacher.action(s,[False]*4)
        self.assertEqual(teacher.phase,'grasp')
        for _ in range(3):teacher.action(s,[False]*4)
        self.assertEqual(teacher.phase,'grasp')
        teacher.action(s,[True,False,False,False]);teacher.action(s,[True,False,False,False])
        self.assertEqual(teacher.phase,'lift')
        _,recover=teacher.action(s,[False]*4)
        self.assertTrue(recover);self.assertEqual(teacher.target,0);self.assertEqual(teacher.phase,'retreat')

    def test_collector_requires_final_success_at_exactly_199_actions(self):
        class Env:
            def __init__(self,final):
                self.final=final;self.num_parcels=4;self.parcels=range(4);self.unwrapped=self
                self.agent=SimpleNamespace(is_grasping=lambda _:torch.tensor([False]))
            def obs(self):
                s=state();s[0]=self.t;return torch.from_numpy(s[None])
            def reset(self,seed):self.t=0;return self.obs(),{}
            def step(self,a):
                self.t+=1
                return self.obs(),0,torch.tensor([False]),torch.tensor([False]),{}
            def evaluate(self):
                # Success earlier in a rollout must not bypass the final score check.
                return dict(success_count=torch.tensor([4 if 20<=self.t<199 else self.final if self.t==199 else 0]))
        class Teacher:
            phase='above'
            def __init__(self,*a):pass
            def action(self,*a):return np.array([0,0,0,1],np.float32),False
        with patch('deadline_collect.DeadlineTeacher',Teacher):
            for final in (3,4):
                env=Env(final);a,r=collect_episode(env,1,1.25)
                self.assertEqual(env.t,199);self.assertEqual(r['accepted'],final==4)
                self.assertEqual(r['first_complete_step'],20)
                self.assertEqual(len(a['obs']),len(a['actions'])+1)

    def test_fast_loader_rejects_long_or_unverified_data_and_samples_all_contacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=dataset(Path(tmp));data=load_data(path);train_ids,val_ids=split_data(data,42)
            self.assertFalse(set(train_ids)&set(val_ids))
            windows=StageWindows(data,train_ids,4,16)
            self.assertEqual({int(data[i]['target'][t]) for i,t in windows.contact},{0,1,2,3})
            self.assertTrue(all(i in train_ids for i,t in windows.contact))
            m=json.loads(path.read_text());m['episodes'][0]['evaluation_actions']=250;path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'deadline'):load_data(path)

    def test_pilot_and_collection_resume_without_repeating_completed_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(gains=[1.25],pilot_episodes=1,pilot_seed_start=100,seed_start=200,
                max_attempts=4,episodes=4,noise_std=0.,noise_probability=0.,seconds_per_call=999)
            job=dict(operation='pilot',folder=tmp,collection_config=cfg,source_sha256='test')
            utils=SimpleNamespace(compose_cfg=lambda x:SimpleNamespace(randomization={}),
                _gym_make=lambda *a:SimpleNamespace(close=lambda:None))
            seen=[]
            def episode(env,seed,gain,*args):
                seen.append(seed)
                return arrays(),dict(seed=seed,gain=gain,accepted=True,correct=4,num_parcels=4,steps=20,
                    evaluation_actions=199,first_complete_step=18,failure=None)
            with patch.dict(sys.modules,{'warehouse_sort.utils':utils}),patch('deadline_collect.collect_episode',side_effect=episode),patch('deadline_collect.sync_from_env'):
                collect(job);collect(job)
                self.assertEqual(seen,[100])
                job['operation']='collect';collect(job);collect(job)
                self.assertEqual(seen,[100,200,201,202,203])
            self.assertEqual(len(load_data(Path(tmp)/'manifest.json')),4)

    def test_model_resume_preserves_better_initial_and_packages_partial_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=dict(CONFIG,profile='smoke',output_root=str(root/'runs'),repo_dir=str(root/'repo'),project_dir=str(ROOT),
                block_iters=4,max_blocks=2,batch_size=4,history=4,width=32,heads=4,layers=1,latent_dim=8,
                validation_batches=1,amp=False,test_record_video=False,record_eval_video=False)
            exp=MediumV2(cfg,source_bundle());exp.connect();exp.repo.mkdir()
            folder=exp.run_dir/'medium';dataset(folder/'collection')
            (folder/'inputs_ready.json').write_text('{}')
            initial=folder/'checkpoints/initial_model.pt';initial.parent.mkdir()
            model=StageACT(exp.model_config('medium'))
            torch.save(dict(format='moveboxes-stage-act-v1',model_config=model.cfg,model=model.state_dict(),step=4000),initial)
            original=exp.training_job
            def job_cpu():return dict(original(),device='cpu')
            fail=[True];eval_seeds=[]
            def run(command,cwd=None,log=None):
                job=json.loads(Path(command[-1]).read_text())
                if command[1]=='stage_train.py':
                    with patch('deadline_train.sync_from_env'):train(job)
                    return
                self.assertEqual(job['max_steps'],200)
                if 'dev_b02' in job['output'] and fail[0]:
                    fail[0]=False;raise RuntimeError('evaluation interrupted')
                eval_seeds.append(job['seeds'])
                score=.5 if 'initial_model' in job['checkpoint'] else .25
                row=dict(sort_accuracy=score,mean_sorted=4*score,all_placed_rate=0,mean_steps=199,mis_sort_rate=0,
                    next_pick=dict(stable_grasp_cycles=1,gripper_sign_reversals=2))
                Path(job['output']).write_text(json.dumps(dict(row,episodes=[dict(row,seed=s) for s in job['seeds']],
                    n_episodes=len(job['seeds']),complete=True,protocol=job)))
            with patch.object(exp,'training_job',side_effect=job_cpu),patch.object(exp,'run',side_effect=run):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):exp.run_blocks()
                exp.run_blocks()
                self.assertEqual(exp._test_candidate('medium')[0],initial)
                final=exp.final_evaluation();self.assertEqual(final['sort_accuracy'],.5)
                self.assertFalse(set(eval_seeds[0])&set(eval_seeds[-1]))
            with zipfile.ZipFile(exp.package()) as archive:
                self.assertFalse(any(Path(n).name.startswith('deadline_') for n in archive.namelist()))
                self.assertNotIn('stage_teacher.py',archive.namelist())
                self.assertIn('stage_policy.py',archive.namelist())

    def test_failed_pilot_blocks_full_collection_and_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(gains=[1.25],pilot_episodes=1,pilot_seed_start=100,seed_start=200,
                max_attempts=4,episodes=4,noise_std=0.,noise_probability=0.,seconds_per_call=999)
            job=dict(operation='collect',folder=tmp,collection_config=cfg,source_sha256='test')
            utils=SimpleNamespace(compose_cfg=lambda x:SimpleNamespace(randomization={}),
                _gym_make=lambda *a:SimpleNamespace(close=lambda:None))
            report=dict(seed=100,gain=1.25,accepted=False,correct=3,first_complete_step=None)
            with patch.dict(sys.modules,{'warehouse_sort.utils':utils}),patch('deadline_collect.collect_episode',return_value=(arrays(),report)) as rollout,patch('deadline_collect.sync_from_env'):
                collect(job)
                self.assertEqual(rollout.call_count,1)
            m=json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertEqual(m['status'],'pilot_failed');self.assertEqual(m['attempts'],[])
            exp=MediumV2(dict(CONFIG,profile='smoke',output_root=tmp,project_dir=str(ROOT)),source_bundle())
            with self.assertRaisesRegex(RuntimeError,'07'):exp.training_job()

    def test_notebook_and_sources_are_complete_and_v1_unchanged(self):
        nb=make_notebook();self.assertEqual(len(nb['cells']),12)
        for i,c in enumerate(nb['cells']):
            self.assertEqual(c['cell_type'],'code');self.assertEqual(c['outputs'],[])
            compile(''.join(c['source']),str(i),'exec')
        source=source_bundle()
        for name,code in source.items():compile(code,name,'exec')
        self.assertIn('from deadline_data import',source['stage_train.py'])
        self.assertNotIn('deadline_data',(ROOT/'ver2/stages/stage_train.py').read_text(encoding='utf-8'))
        self.assertNotEqual(CONFIG['run_name'],CONFIG['source_run_name'])
        self.assertEqual(CONFIG['block_iters']*CONFIG['max_blocks'],3000)

    def test_bootstrap_executes_and_keeps_fresh_git_commit_and_v2_defaults(self):
        # Execute the actual generated cell; compilation alone cannot catch a missing CFG key.
        import contextlib, io
        nb=make_notebook()
        fresh_commit='a'*40
        for stale in (None,'b'*40):
            with self.subTest(previous_commit=stale),tempfile.TemporaryDirectory() as tmp:
                scope={}
                exec(''.join(nb['cells'][0]['source']),scope)
                scope['CFG'].update(project_dir=str(ROOT),output_root=tmp,profile='smoke',batch_size=24)
                if stale:scope['CFG']['project_commit']=stale
                def git_output(command,**kwargs):
                    if command==['git','remote','get-url','origin']:
                        return 'https://github.com/SongYunu/moveBoxes.git\n'
                    self.assertEqual(command,['git','rev-parse','HEAD'])
                    return fresh_commit+'\n'
                previous_path=list(sys.path)
                try:
                    with patch('subprocess.run') as run,patch('subprocess.check_output',side_effect=git_output), \
                         patch('importlib.reload',side_effect=lambda module:module),contextlib.redirect_stdout(io.StringIO()):
                        exec(''.join(nb['cells'][1]['source']),scope)
                    self.assertTrue(all(c.args[0][0]=='git' for c in run.call_args_list))
                    self.assertEqual(scope['CFG']['project_commit'],fresh_commit)
                    self.assertEqual(scope['experiment'].cfg['project_commit'],fresh_commit)
                    self.assertEqual(scope['experiment'].cfg['batch_size'],24)
                    self.assertEqual(scope['experiment'].cfg['position_noise'],0.)
                    self.assertEqual(scope['experiment'].cfg['warmup_steps'],50)
                    self.assertFalse(scope['experiment'].connected)
                finally:
                    sys.path[:]=previous_path
