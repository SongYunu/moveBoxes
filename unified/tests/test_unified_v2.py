import contextlib,io,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_unified import ROOT,dataset,state
import torch
from stage_model import StageACT
from unified_v2 import UnifiedV2,source_bundle
from build_unified_v2_notebook import CONFIG,make_notebook
from unified_resume import resume_matches


class UnifiedV2Tests(unittest.TestCase):
    def test_bootstrap_and_native_evaluator(self):
        nb=make_notebook();scope={};exec(''.join(nb['cells'][0]['source']),scope)
        self.assertEqual(scope['CFG']['easy_source_run_name'],'moveboxes_easy_lab_v1')
        self.assertNotEqual(scope['CFG']['run_name'],'moveboxes_unified_curriculum_v1')
        scope['CFG'].update(project_dir=str(ROOT),profile='smoke')
        def git(command,**kwargs):return 'https://github.com/SongYunu/moveBoxes.git\n' if 'remote' in command else 'b'*40+'\n'
        with patch('subprocess.run'),patch('subprocess.check_output',side_effect=git),patch('importlib.reload',side_effect=lambda m:m),contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(nb['cells'][1]['source']),scope)
        self.assertIsInstance(scope['experiment'],UnifiedV2)
        for cell in nb['cells']:compile(''.join(cell['source']),'cell','exec')
        for name,code in source_bundle().items():compile(code,name,'exec')
        self.assertIn('from stage_policy import load_stage',source_bundle()['unified_native_eval.py'])

    def test_focus_changes_keep_optimizer_and_cumulative_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=[dataset(root,'easy',54)]
            for name,code in source_bundle().items():(root/name).write_text(code,encoding='utf-8')
            arch=dict(state_dim=90,history=4,chunk_size=4,width=32,heads=4,layers=1,latent_dim=4)
            initial=root/'initial.pt';model=StageACT(arch)
            torch.save(dict(format='moveboxes-stage-act-v1',model_config=arch,model=model.state_dict(),step=0),initial)
            manifest=root/'data.json';manifest.write_text(json.dumps(dict(sources=sources,active='easy',focus={})))
            cfg=dict(seed=42,batch_size=4,lr=1e-4,warmup_steps=1,total_iters=4,save_freq=2,validation_batches=1,amp=False,
                position_noise=0.,console_interval_seconds=999,kl_weight=0.,stage_loss_weight=.3,gate_loss_weight=.3,
                replay_fraction=.3,speed_bonus=.5,focus_fraction=.7,action_training_mode='prior')
            job=dict(folder=str(root/'run'),data=str(manifest),model_config=arch,train_config=cfg,policy_config={},
                warm_start=str(initial),source_sha256='test-v2',device='cpu',stop_at=2)
            path=root/'job.json'
            def run():
                path.write_text(json.dumps(job));result=subprocess.run([sys.executable,str(root/'stage_train.py'),str(path)],cwd=root,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return torch.load(root/'run/checkpoints/latest.pt',weights_only=True)
            first=run();self.assertEqual(first['step'],2)
            manifest.write_text(json.dumps(dict(sources=sources,active='easy',focus=dict(parcel_ids=[0],stage=0))))
            job['stop_at']=4;second=run();self.assertEqual(second['step'],4)
            self.assertTrue(any(int(v['step'])==4 for v in second['optimizer']['state'].values()))
            self.assertNotEqual(first['signature']['data_sha256'],second['signature']['data_sha256'])
            self.assertEqual(first['signature']['curriculum_dataset_sha256'],second['signature']['curriculum_dataset_sha256'])
            changed=dict(second['signature'],warm_start_sha256='different-model')
            self.assertFalse(resume_matches(second['signature'],changed))
            self.assertFalse(resume_matches(second['signature'],dict(second['signature'],curriculum_dataset_sha256='different-data')))

    def test_rejected_learner_is_reused_instead_of_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obj=UnifiedV2(dict(CONFIG,output_root=tmp),{});obj.run_dir=root
            latest=root/'easy/learner_01/checkpoints/latest.pt';latest.parent.mkdir(parents=True);latest.write_bytes(b'present')
            state=dict(learners={'easy':dict(folder='easy/learner_01',start_round=1,anchor='easy/initial.pt',focus={'stage':2})})
            with patch.object(obj,'_train_job',return_value={}) as make_job:
                entry,job=obj._learner_job(state,'easy',2,root/'selected_best.pt',{})
            self.assertEqual(job['stop_at'],1000)
            self.assertEqual(make_job.call_args.args[1],root/'easy/learner_01')
            self.assertEqual(make_job.call_args.args[2],root/'easy/initial.pt')
            self.assertEqual(make_job.call_args.args[3],{'stage':2})


if __name__=='__main__':unittest.main()
