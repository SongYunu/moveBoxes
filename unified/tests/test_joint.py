import contextlib,io,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_unified import ROOT,dataset,state
import sys,torch
from unified_data import load_curriculum
from joint_data import JointWindows
from joint_experiment import JointExperiment,joint_quality,source_bundle
from build_joint_notebook import CONFIG,make_notebook
from marso_experiment import save_json,read_json
from object_model import ObjectACT,FORMAT,canonical_state
from object_policy import load_stage


class JointTests(unittest.TestCase):
    def test_equal_task_batches_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=[dataset(Path(tmp),level,dim) for level,dim in zip(('easy','medium','hard'),(54,72,90))]
            items,train,valid=load_curriculum(sources,42)
            for level in train:self.assertFalse(set(train[level])&set(valid[level]))
            windows=JointWindows(items,train,4,4)
            generator=torch.Generator().manual_seed(42)
            for _ in range(30):
                counts=windows.counts(32,generator)
                self.assertEqual(sum(counts),32);self.assertEqual(sorted(counts),[10,11,11])
            batch=windows.batch(32,generator,'cpu')
            # The parcel tag slots distinguish the three original difficulties.
            n=(batch['obs'][:,-1,68:80].reshape(32,6,2).sum(-1)>0).sum(-1)
            self.assertEqual(sorted([(n==i).sum().item() for i in (2,4,6)]),[10,11,11])

    def test_bootstrap_has_no_pretrained_source_and_no_gates(self):
        nb=make_notebook();scope={};exec(''.join(nb['cells'][0]['source']),scope)
        self.assertNotIn('medium_source_run_name',scope['CFG'])
        scope['CFG']['project_dir']=str(ROOT)
        def git(cmd,**kwargs):return 'https://github.com/SongYunu/moveBoxes.git\n' if 'remote' in cmd else 'e'*40+'\n'
        with patch('subprocess.run'),patch('subprocess.check_output',side_effect=git),patch('importlib.reload',side_effect=lambda m:m),contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(nb['cells'][1]['source']),scope)
        self.assertIsInstance(scope['experiment'],JointExperiment)
        for c in nb['cells']:
            code=''.join(c['source']);compile(code,'cell','exec');self.assertNotIn('train_level(',code)
        for name,code in source_bundle().items():compile(code,name,'exec')

    def test_scratch_job_has_no_warmstart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obj=JointExperiment(dict(CONFIG,output_root=tmp),{});obj.run_dir=root
            state=dict(segment=None,checkpoint=None,completed_updates=0)
            with patch.object(obj,'_dataset',return_value=dict(mode='joint',sources=[])):
                job=obj._make_job(state)
            self.assertIsNone(job['warm_start']);self.assertFalse(job['resume_joint'])
            self.assertEqual(job['stop_at'],2000);self.assertEqual(job['train_config']['total_iters'],30000)

    def test_cpu_scratch_training_and_exact_local_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,code in source_bundle().items():(root/name).write_text(code,encoding='utf-8')
            sources=[dataset(root,level,dim) for level,dim in zip(('easy','medium','hard'),(54,72,90))]
            manifest=root/'data.json';save_json(manifest,dict(mode='joint',sources=sources))
            arch=dict(architecture='object-attention-v1',state_dim=90,history=4,chunk_size=4,width=32,heads=4,layers=1,spatial_layers=1)
            cfg=dict(seed=42,batch_size=6,lr=1e-4,warmup_steps=1,total_iters=4,save_freq=2,validation_batches=1,amp=False,
                position_noise=0.,console_interval_seconds=999,kl_weight=0.,stage_loss_weight=.3,gate_loss_weight=.3,action_training_mode='prior')
            job=dict(folder=str(root/'run'),data=str(manifest),model_config=arch,train_config=cfg,policy_config={},
                warm_start=None,resume_joint=False,source_sha256='joint-test',device='cpu',stop_at=2)
            def run():
                save_json(root/'job.json',job)
                result=subprocess.run([sys.executable,str(root/'stage_train.py'),str(root/'job.json')],cwd=root,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return torch.load(root/'run/checkpoints/latest.pt',weights_only=True)
            first=run();self.assertEqual(first['step'],2);self.assertIsNone(first['signature']['warm_start_sha256'])
            self.assertEqual(first['format'],FORMAT)
            job['stop_at']=4;second=run();self.assertEqual(second['step'],4)
            self.assertTrue(any(int(v['step'])==4 for v in second['optimizer']['state'].values()))
            report=read_json(root/'run/resource_report.json');self.assertEqual(set(report['train_episodes']),{'easy','medium','hard'})
            self.assertEqual(report['updates_this_call'],2)
            # The exported policy must load independently with the same weights for all levels.
            (root/'deployment_check.py').write_text('''
import torch
from types import SimpleNamespace
from colab_policy import load_stage
from object_runtime_check import sample_state
for dim in (54,72,90):
    obs=sample_state(dim,1)
    policy=load_stage('run/checkpoints/latest.pt',obs,SimpleNamespace(shape=(4,)),'cpu')
    action=policy.act(obs)
    assert torch.isfinite(action).all() and action.shape==(1,4)
    policy.act(obs);policy.reset()
    torch.testing.assert_close(action,policy.act(obs))
''',encoding='utf-8')
            result=subprocess.run([sys.executable,'deployment_check.py'],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_set_invariance_and_padding(self):
        torch.set_num_threads(2);torch.manual_seed(4)
        cfg=dict(architecture='object-attention-v1',state_dim=90,history=4,chunk_size=4,width=32,heads=4,layers=1)
        model=ObjectACT(cfg).eval()
        # Different per-proprioception statistics cannot introduce parcel-slot identity.
        model.obs_mean.copy_(torch.randn(26));model.obs_std.copy_(torch.rand(26)+.1)
        obs=canonical_state(torch.tensor(state(72,4)))[None]
        obs[...,86:90]=torch.tensor([1.,0.,0.,1.])
        perm=torch.tensor([3,5,0,4,2,1]);swapped=obs.clone()
        swapped[...,26:68]=obs[...,26:68].reshape(1,4,6,7)[...,perm,:].flatten(-2)
        swapped[...,68:80]=obs[...,68:80].reshape(1,4,6,2)[...,perm,:].flatten(-2)
        swapped[...,80:86]=obs[...,80:86].reshape(1,4,2,3).flip(-2).flatten(-2)
        swapped[...,86:90]=obs[...,86:90].reshape(1,4,2,2).flip(-2).flatten(-2)
        absent=obs.clone();absent[...,54:68]=torch.randn(1,4,14)*100
        flipped=obs.clone();poses=flipped[...,26:68].reshape(1,4,6,7);poses[...,3:]*=-1
        with torch.no_grad():
            expected=model(obs)[0]
            for variant in (swapped,absent,flipped):torch.testing.assert_close(expected,model(variant)[0],atol=2e-6,rtol=1e-5)
            # Geometry remains observable; invariance is not achieved by ignoring objects.
            shifted=obs.clone();shifted[...,26]+=.07
            self.assertGreater((expected-model(shifted)[0]).abs().max().item(),1e-5)
            torch.testing.assert_close(expected,model(obs,torch.ones(1,dtype=torch.long),torch.ones(1,dtype=torch.long))[0])

    def test_episode_chunks_cross_phase_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=[dataset(Path(tmp),level,dim) for level,dim in zip(('easy','medium','hard'),(54,72,90))]
            items,train,valid=load_curriculum(sources,42)
            for item in items:item['stage']=torch.arange(len(item['actions']))%4
            windows=JointWindows(items,train,4,4)
            batch=windows.batch(30,torch.Generator().manual_seed(42),'cpu')
            self.assertTrue((batch['mask'].sum(-1)==4).any())

    def test_package_is_self_contained(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obj=JointExperiment(dict(CONFIG,output_root=tmp),source_bundle());obj.run_dir=root
            model=ObjectACT(obj.model_config('easy'))
            checkpoint=root/'best.pt';torch.save(dict(format=FORMAT,model_config=model.cfg,model=model.state_dict()),checkpoint)
            from marso_experiment import digest
            save_json(root/'curriculum.json',dict(checkpoint='best.pt',checkpoint_sha256=digest(checkpoint)))
            with patch.object(obj,'_ready'),patch.object(obj,'sync_common'):
                archive=obj.package()
            unpack=root/'unpack'
            with zipfile.ZipFile(archive) as z:z.extractall(unpack)
            submission=read_json(unpack/'submission.yaml')
            self.assertEqual({row['checkpoint'] for row in submission['state']['levels'].values()},{'checkpoints/model.pt'})
            code="from colab_policy import load_policy; import torch; from types import SimpleNamespace; p=load_policy('checkpoints/model.pt',torch.zeros(1,54),SimpleNamespace(shape=(4,)),'cpu'); assert p.act(torch.zeros(1,54)).shape==(1,4)"
            result=subprocess.run([sys.executable,'-c',code],cwd=unpack,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_backup_contains_one_model_and_joint_metadata_in_common_scope(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obj=JointExperiment(dict(CONFIG,output_root=tmp),{})
            obj.run_dir=root;obj.session=root/'sessions/current';obj.session.mkdir(parents=True)
            obj.connected=True;obj.remote_enabled=True;obj.store=Mock()
            selected=root/'joint/candidates/best.pt';selected.parent.mkdir(parents=True);selected.write_bytes(b'best')
            (selected.parent/'other.pt').write_bytes(b'other')
            save_json(root/'curriculum.json',dict(checkpoint='joint/candidates/best.pt',checkpoint_sha256='test'))
            metadata=root/'joint/segment_000000/job.json';save_json(metadata,dict(example=True))
            obj.sync_common()
            args=obj.store.sync.call_args.args
            self.assertEqual(args[1],'common');self.assertIn(metadata,args[2])
            self.assertEqual([p for p in args[2] if p.suffix=='.pt'],[selected])

    def test_object_observer_does_not_require_stage_policy(self):
        from object_observer import Observer
        cfg=dict(architecture='object-attention-v1',state_dim=90,history=4,chunk_size=4,width=32,heads=4,layers=1)
        from object_policy import ObjectPolicy
        observer=Observer(ObjectPolicy(ObjectACT(cfg)))
        obs=torch.tensor(state(54,1));action=observer.act(obs)
        self.assertEqual(action.shape,(1,4));self.assertEqual(observer.report()['observed_steps'],1)
        self.assertNotIn('stages',observer.report())
        observer.reset();self.assertEqual(observer.report()['observed_steps'],0)

    def test_quality_requires_all_tasks(self):
        metrics={k:dict(complete=True,sort_accuracy=1.,episodes=[dict(seed=1,mean_sorted=n)]) for k,n in zip(('easy','medium','hard'),(2,4,6))}
        self.assertEqual(joint_quality(metrics)['macro_score'],1.)
        del metrics['hard']
        with self.assertRaises(ValueError):joint_quality(metrics)


if __name__=='__main__':unittest.main()
