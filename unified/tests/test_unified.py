import contextlib,io,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('unified/code','hard/code','ver2/stages','ver2','')]
from stage_model import StageACT
from unified_policy import canonical_state,load_stage
from unified_data import load_curriculum,CurriculumWindows
from unified_experiment import UnifiedExperiment,source_bundle
from build_unified_notebook import make_notebook,CONFIG
from hard_transfer import transplant
from hard_progress import ParcelProgress,paired_comparison
from hard_observer import instrument_observer
from marso_experiment import digest


def state(dim=72,steps=13):
    n=(dim-36)//9;obs=np.zeros((steps,dim),np.float32);obs[:,18:21]=[0,0,.17]
    for i in range(n):obs[:,26+7*i:29+7*i]=[-.05,.04*i,.03];obs[:,29+7*i]=1;obs[:,26+7*n+2*i+(i%2)]=1
    obs[:,26+9*n:32+9*n]=[0,-.36,0,0,.36,0];return obs


def dataset(folder,level,dim):
    path=folder/(level+'.h5')
    with h5py.File(path,'w') as f:
        for i in range(5):
            g=f.create_group(f'traj_{i}');obs=state(dim);obs[:,0]=i*.1
            actions=np.zeros((12,4),np.float32);actions[:,3]=1
            g.create_dataset('obs',data=obs);g.create_dataset('actions',data=actions)
    path.with_suffix('.json').write_text(json.dumps(dict(episodes=[dict(episode_id=i,info=dict(success=i%2==0)) for i in range(5)])))
    return dict(level=level,data=str(path),num_demos=None)


class UnifiedTests(unittest.TestCase):
    def test_shared_model_actions_all_shapes_and_medium_equivalence(self):
        cfg=dict(state_dim=72,history=4,chunk_size=4,width=32,heads=4,layers=1,latent_dim=4)
        donor=StageACT(cfg).eval();hard=dict(cfg,state_dim=90)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.pt';target=Path(tmp)/'model.pt'
            torch.save(dict(format='moveboxes-stage-act-v1',model_config=cfg,model=donor.state_dict(),step=7),source)
            transplant(source,hard,torch.zeros(126),torch.ones(126),target)
            model=StageACT(hard).eval();model.load_state_dict(torch.load(target,weights_only=True)['model'])
            obs=torch.tensor(state(72,4))[None];phase=torch.zeros(1,dtype=torch.long)
            with torch.no_grad():
                self.assertTrue(torch.allclose(donor(obs,phase,phase)[0],model(canonical_state(obs),phase,phase)[0],atol=1e-6))
            before=target.read_bytes();transplant(source,hard,torch.zeros(126),torch.ones(126),target);self.assertEqual(before,target.read_bytes())
            for dim in (54,72,90):
                raw=torch.tensor(state(dim,1));policy=load_stage(target,raw,type('A',(),{'shape':(4,)})(),'cpu',ensemble_window=1)
                self.assertEqual(policy.act(raw).shape,(1,4));policy.reset();self.assertEqual(policy.act(raw).shape,(1,4))

    def test_split_replay_and_expert_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=[dataset(Path(tmp),'easy',54),dataset(Path(tmp),'medium',72)]
            trajectories,train,valid=load_curriculum(sources,42)
            for level in train:self.assertFalse(set(train[level]) & set(valid[level]))
            self.assertTrue(all(t['obs'].shape[-1]==90 for t in trajectories))
            windows=CurriculumWindows(trajectories,train,4,4,'medium',dict(parcel_ids=[0],stage=0),replay_fraction=.5)
            self.assertIsNotNone(windows.focus)
            self.assertTrue(all(i in train['medium'] for i,t in windows.focus.indices))
            batch=windows.batch(8,torch.Generator().manual_seed(1),'cpu')
            self.assertEqual(batch['obs'].shape,(8,4,90))
            self.assertTrue(torch.isfinite(batch['obs']).all())

    def test_bootstrap_and_difficulty_gate(self):
        nb=make_notebook();scope={};exec(''.join(nb['cells'][0]['source']),scope)
        scope['CFG'].update(project_dir=str(ROOT),profile='smoke')
        def git(command,**kwargs):return 'https://github.com/SongYunu/moveBoxes.git\n' if 'remote' in command else 'a'*40+'\n'
        with patch('subprocess.run'),patch('subprocess.check_output',side_effect=git),patch('importlib.reload',side_effect=lambda m:m),contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(nb['cells'][1]['source']),scope)
        obj=scope['experiment'];self.assertIsInstance(obj,UnifiedExperiment)
        for cell in nb['cells']:compile(''.join(cell['source']),'cell','exec')
        for name,code in source_bundle().items():compile(code,name,'exec')
        with patch.object(obj,'_ready'),patch.object(obj,'_stage_helpers'),patch.object(obj,'_current',return_value=(dict(passed=[]),Path('fake.pt'))),patch.object(obj,'_evaluate') as evaluate,contextlib.redirect_stdout(io.StringIO()):
            obj.train_level('hard');evaluate.assert_not_called()
        self.assertFalse(set(obj.seeds(True))&set(obj.test_seeds()))
        self.assertFalse(set(obj.seeds(True))&set(obj.seeds(False)))

    def test_cpu_staged_training_and_resource_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=[dataset(root,'easy',54),dataset(root,'medium',72)]
            arch=dict(state_dim=90,history=4,chunk_size=4,width=32,heads=4,layers=1,latent_dim=4)
            initial=root/'initial.pt';model=StageACT(arch)
            torch.save(dict(format='moveboxes-stage-act-v1',model_config=arch,model=model.state_dict(),step=0),initial)
            for name,code in source_bundle().items():(root/name).write_text(code,encoding='utf-8')
            manifest=root/'data.json';manifest.write_text(json.dumps(dict(sources=sources,active='easy',focus=dict(parcel_ids=[0],stage=0))))
            cfg=dict(seed=42,batch_size=4,lr=1e-4,warmup_steps=1,total_iters=2,save_freq=1,validation_batches=1,amp=False,
                position_noise=0.,console_interval_seconds=999,kl_weight=0.,stage_loss_weight=.3,gate_loss_weight=.3,replay_fraction=.3,speed_bonus=.5,action_training_mode='prior')
            job=dict(folder=str(root/'run'),data=str(manifest),model_config=arch,train_config=cfg,policy_config={},warm_start=str(initial),source_sha256='test',device='cpu')
            path=root/'job.json';path.write_text(json.dumps(job))
            result=subprocess.run([sys.executable,str(root/'stage_train.py'),str(path)],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            saved=torch.load(root/'run/checkpoints/latest.pt',weights_only=True);self.assertEqual(saved['step'],2)
            report=json.loads((root/'run/resource_report.json').read_text());self.assertGreater(report['elapsed_seconds'],0)
            self.assertEqual(set(report['training_episodes']),{'easy','medium'})

    def test_repeat_grasp_is_not_a_score_improvement(self):
        monitor=ParcelProgress()
        for i,ids in enumerate(([1],[],[1],[],[1])):monitor.record(ids,[],i)
        self.assertEqual(monitor.report()['repeat_grasp_events'],2)
        metrics=dict(complete=True,protocol=dict(level='easy',max_steps=200),episodes=[dict(seed=1,sort_accuracy=0)])
        self.assertFalse(paired_comparison(metrics,metrics)['promote'])

    def test_dataset_identity_and_confirmation_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=dataset(Path(tmp),'easy',54)
            source.update(data_sha256=digest(source['data']),metadata_sha256=digest(Path(source['data']).with_suffix('.json')))
            load_curriculum([source],42)
            Path(source['data']).with_suffix('.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'metadata changed'):load_curriculum([source],42)
        with self.assertRaisesRegex(ValueError,'Confirmation seeds'):
            UnifiedExperiment(dict(CONFIG,confirmation_seed_start=CONFIG['tuning_seed_start']),{})

    def test_observer_does_not_change_actions_and_captures_terminal_sort(self):
        class Parent:
            def reset(self):self.decisions=[]
            def act(self,obs,deterministic=True):
                self.decisions.append(dict(stage=0));return obs
            def report(self):return {}
        observer=instrument_observer(Parent)();observer.reset()
        from types import SimpleNamespace
        env=SimpleNamespace(parcels=[0,1],agent=SimpleNamespace(is_grasping=lambda p:torch.tensor([p==0])),_placed_correct=torch.tensor([[False,False]]))
        observer.env=SimpleNamespace(unwrapped=env)
        action=torch.ones(1,4);self.assertIs(observer.act(action),action)
        env._placed_correct[0,0]=True;observer.capture()
        self.assertEqual(observer.report()['parcel_progress']['sorted_parcel_first_actions'],{'0':1})

    def test_package_contains_one_shared_checkpoint(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'selected.pt';checkpoint.write_bytes(b'fixture')
            obj=UnifiedExperiment(dict(CONFIG,output_root=tmp),source_bundle());obj.run_dir=root
            state=dict(checkpoint='selected.pt',checkpoint_sha256=digest(checkpoint),passed=[],rounds={},history=[])
            with patch.object(obj,'_ready'),patch.object(obj,'_current',return_value=(state,checkpoint)),patch.object(obj,'sync_common'),contextlib.redirect_stdout(io.StringIO()):
                path=obj.package()
            with zipfile.ZipFile(path) as archive:
                self.assertEqual([n for n in archive.namelist() if n.endswith('.pt')],['checkpoints/model.pt'])
                submission=json.loads(archive.read('submission.yaml'))
                self.assertEqual({v['checkpoint'] for v in submission['state']['levels'].values()},{'checkpoints/model.pt'})


if __name__=='__main__':unittest.main()
