import contextlib,io,json,sys,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
from test_unified import ROOT,dataset,state
import numpy as np,torch
from stage_model import StageACT
from unified_medium import UnifiedMedium,medium_selection,source_bundle
from build_unified_medium_notebook import CONFIG,make_notebook
from marso_experiment import save_json,read_json,digest


def metrics(score=0,counts=None,steps=50):
    counts=counts or [score*4,score*4]
    rows=[dict(seed=i,mean_sorted=c,sort_accuracy=c/4,next_pick=dict(parcel_progress=dict(sorted_parcel_first_actions={'0':steps}),final_stage=0)) for i,c in enumerate(counts)]
    return dict(complete=True,sort_accuracy=sum(r['sort_accuracy'] for r in rows)/len(rows),protocol=dict(level='medium',max_steps=200),episodes=rows)


def deadlines(root):
    entries=[]
    for i,n in enumerate((8,10,12,14,16)):
        path=root/f'demo_{i}.npz'
        np.savez(path,obs=state(72,n+1),actions=np.zeros((n,4),np.float32),previous_stage=np.zeros(n,np.int64),
            stage=np.zeros(n,np.int64),gate=np.zeros(n,np.int64),target=np.zeros(n,np.int64),perturbed=np.zeros(n,bool))
        correct=3 if i==0 else 4
        entries.append(dict(file=path.name,sha256=digest(path),accepted=True,correct=correct,evaluation_actions=199,
            minimum_correct=3,failure=None,fully_successful=correct==4,completion_steps={str(correct):n}))
    path=root/'manifest.json';save_json(path,dict(version='curriculum-v21',complete=True,minimum_correct=3,episodes=entries))
    return dict(level='medium',data=str(path),format='curriculum-v21',manifest_sha256=digest(path))


class MediumAdaptationTests(unittest.TestCase):
    def test_easy_evaluation_uses_medium_winner_and_labels_initial_model(self):
        obj=UnifiedMedium(CONFIG,{})
        initial=dict(checkpoint='easy/ref.pt',easy_reference='easy/ref.pt',history=[])
        with patch.object(obj,'_ready'),patch.object(obj,'_current',return_value=(initial,Path('initial.pt'))),patch.object(obj,'test') as evaluate,contextlib.redirect_stdout(io.StringIO()):
            obj.test_medium_on_easy();evaluate.assert_not_called()
        selected=dict(checkpoint='medium/candidate.pt',easy_reference='easy/ref.pt',history=[dict(level='medium',accepted=True)])
        with tempfile.TemporaryDirectory() as tmp:
            candidate=Path(tmp)/'candidate.pt';candidate.write_bytes(b'medium-winner')
            with patch.object(obj,'_ready'),patch.object(obj,'_current',return_value=(selected,candidate)),patch.object(obj,'test') as evaluate,contextlib.redirect_stdout(io.StringIO()):
                obj.test_medium_on_easy();evaluate.assert_called_once_with('easy')
                self.assertEqual(obj._test_candidate('easy')[0],obj._test_candidate('medium')[0])

    def test_notebook_bootstrap_and_budget(self):
        nb=make_notebook();scope={};exec(''.join(nb['cells'][0]['source']),scope)
        scope['CFG'].update(project_dir=str(ROOT),profile='smoke')
        def git(cmd,**kw):return 'https://github.com/SongYunu/moveBoxes.git\n' if 'remote' in cmd else 'c'*40+'\n'
        with patch('subprocess.run'),patch('subprocess.check_output',side_effect=git),patch('importlib.reload',side_effect=lambda m:m),contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(nb['cells'][1]['source']),scope)
        self.assertIsInstance(scope['experiment'],UnifiedMedium)
        self.assertEqual(CONFIG['block_iters']*CONFIG['max_blocks']['medium'],20000)
        self.assertFalse(any('train_level("easy")' in ''.join(c['source']) for c in nb['cells']))
        for c in nb['cells']:compile(''.join(c['source']),'cell','exec')
        for name,code in source_bundle().items():compile(code,name,'exec')

    def test_success_speed_never_rewards_failed_fast_episode(self):
        self.assertFalse(medium_selection(metrics(0,steps=100),metrics(0,steps=1))[0])
        self.assertTrue(medium_selection(metrics(counts=[4,0],steps=100),metrics(counts=[4,0],steps=80))[0])
        self.assertFalse(medium_selection(metrics(counts=[4,2],steps=100),metrics(counts=[4,0],steps=1))[0])

    def test_zero_scores_do_not_stop_training_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=UnifiedMedium(dict(CONFIG,output_root=tmp,max_blocks=dict(easy=1,medium=4,hard=1)),{})
            root=Path(tmp);obj.run_dir=root
            initial=root/'easy.pt';initial.write_bytes(b'initial')
            save_json(root/'curriculum.json',dict(checkpoint='easy.pt',checkpoint_sha256=digest(initial),rounds={},history=[],learners={},passed=[]))
            jobs=[]
            def learner(state,level,number,current,baseline):
                job=dict(folder=str(root/'medium/learner'),stop_at=number*1000);jobs.append(job);return {},job
            def run(*args,**kwargs):
                path=root/'medium/learner/checkpoints/latest.pt';path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(format='test',model_config={},model={},step=jobs[-1]['stop_at']),path)
            with patch.object(obj,'_ready'),patch.object(obj,'_stage_helpers'),patch.object(obj,'_evaluate',return_value=metrics()),patch.object(obj,'_learner_job',side_effect=learner),patch.object(obj,'run',side_effect=run),patch.object(obj,'persist_operation',side_effect=lambda level:contextlib.nullcontext()),contextlib.redirect_stdout(io.StringIO()):
                obj.train_medium()
            saved=read_json(root/'curriculum.json')
            self.assertEqual(saved['rounds']['medium'],4)
            self.assertEqual(saved['checkpoint'],'easy.pt')

    def test_mixed_original_and_deadline_demos_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,code in source_bundle().items():(root/name).write_text(code,encoding='utf-8')
            sources=[dataset(root,'easy',54),dataset(root,'medium',72),deadlines(root)]
            arch=dict(state_dim=90,history=4,chunk_size=4,width=32,heads=4,layers=1,latent_dim=4)
            initial=root/'initial.pt';model=StageACT(arch)
            torch.save(dict(format='moveboxes-stage-act-v1',model_config=arch,model=model.state_dict(),step=0),initial)
            path=root/'data.json';save_json(path,dict(sources=sources,active='medium',focus=dict(parcel_ids=[0],stage=0)))
            cfg=dict(seed=42,batch_size=4,lr=1e-4,warmup_steps=1,total_iters=2,save_freq=2,validation_batches=1,amp=False,
                position_noise=0.,console_interval_seconds=999,kl_weight=0.,stage_loss_weight=.3,gate_loss_weight=.3,
                replay_fraction=.3,speed_bonus=.5,focus_fraction=.7,contact_sampling=True,action_training_mode='prior')
            job=dict(folder=str(root/'run'),data=str(path),model_config=arch,train_config=cfg,policy_config={},warm_start=str(initial),source_sha256='medium-test',device='cpu')
            save_json(root/'job.json',job)
            result=subprocess.run([sys.executable,str(root/'stage_train.py'),str(root/'job.json')],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            audit=read_json(root/'run/data_audit.json');self.assertEqual(audit['recovery_episodes'],5)
            self.assertFalse(set(audit['train_trajectories']) & set(audit['validation_trajectories']))

    def test_easy_regression_does_not_reject_medium_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=UnifiedMedium(dict(CONFIG,output_root=tmp,easy_check_interval=1,max_blocks=dict(easy=1,medium=1,hard=1)),{})
            root=Path(tmp);obj.run_dir=root;reference=root/'easy.pt';reference.write_bytes(b'easy-original')
            save_json(root/'curriculum.json',dict(checkpoint='easy.pt',checkpoint_sha256=digest(reference),rounds={},history=[],learners={},passed=[]))
            def learner(*args):return {},dict(folder=str(root/'medium/learner'),stop_at=1000)
            def run(*args,**kwargs):
                path=root/'medium/learner/checkpoints/latest.pt';path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(format='test',model_config={},model={},step=1000),path)
            def evaluate(level,checkpoint,label):return metrics(.5) if label=='candidate_01' else metrics(0)
            with patch.object(obj,'_ready'),patch.object(obj,'_stage_helpers'),patch.object(obj,'_evaluate',side_effect=evaluate),patch.object(obj,'_learner_job',side_effect=learner),patch.object(obj,'run',side_effect=run),patch.object(obj,'persist_operation',side_effect=lambda level:contextlib.nullcontext()),contextlib.redirect_stdout(io.StringIO()):
                obj.train_medium()
            record=read_json(root/'curriculum.json')
            self.assertEqual(record['checkpoint'],'medium/candidates/round_01.pt')
            self.assertEqual(record['history'][0]['easy_score'],0)
            self.assertTrue(record['history'][0]['accepted'])
            self.assertEqual(reference.read_bytes(),b'easy-original')


if __name__=='__main__':unittest.main()
