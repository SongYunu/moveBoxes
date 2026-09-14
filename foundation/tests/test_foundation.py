import json,sys,tempfile,unittest
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'foundation/code'
sys.path.insert(0,str(CODE));sys.path.insert(0,str(ROOT))
from foundation_common import LEVELS,RGBData,action_to_env,normalize,read,write
from foundation_experiment import FoundationExperiment


class FoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        roots=[ROOT.parent/level for level in LEVELS]
        cls.have_data=all((root/'trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5').exists() for root in roots)
        cls.data=RGBData(ROOT.parent,17) if cls.have_data else None

    @classmethod
    def tearDownClass(cls):
        if cls.data:cls.data.close()

    def test_sources_compile(self):
        for path in CODE.glob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')

    def test_notebooks_are_code_only_and_compile(self):
        for name,backend in (('moveboxes_smolvla_colab.ipynb','smol'),('moveboxes_octo_colab.ipynb','octo')):
            book=json.loads((ROOT/'foundation/notebooks'/name).read_text(encoding='utf-8'))
            self.assertEqual(len(book['cells']),11)
            self.assertTrue(all(c['cell_type']=='code' for c in book['cells']))
            source=''.join(book['cells'][0]['source'])
            self.assertIn(f"'backend': '{backend}'",source)
            for i,c in enumerate(book['cells']):compile(''.join(c['source']),f'{name}:{i+1}','exec')
        combined=json.loads((ROOT/'foundation/notebooks/moveboxes_foundation_both_colab.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(len(combined['cells']),11)
        self.assertTrue(all(c['cell_type']=='code' for c in combined['cells']))
        source='\n'.join(''.join(c['source']) for c in combined['cells'])
        self.assertIn("BACKENDS = ['smol', 'octo']",source)
        self.assertIn("run_each('train')",source)
        for i,c in enumerate(combined['cells']):compile(''.join(c['source']),f'combined:{i+1}','exec')

    def test_pins_and_locked_requirements(self):
        common=(CODE/'foundation_common.py').read_text(encoding='utf-8')
        for value in ('c83c3163b8ca9b7e67c509fffd9121e66cb96205','dc9aa3019f764726c770814b27e4ab0fc6e32a58','241fb3514b7c40957a86d869fecb7c7fc353f540'):
            self.assertIn(value,common)
        requirements=ROOT/'foundation/requirements'
        self.assertIn('jaxlib==0.4.20+cuda12.cudnn89',(requirements/'octo.lock').read_text())
        self.assertIn('mplib==0.1.1',(requirements/'simulator.lock').read_text())
        self.assertIn('lerobot==0.4.3',(requirements/'smolvla.lock').read_text())
        experiment=(CODE/'foundation_experiment.py').read_text(encoding='utf-8')
        self.assertIn("SCORE_WEIGHTS={'easy':.2,'medium':.3,'hard':.5}",experiment)

    def test_json_atomic_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'nested/value.json';write(path,{'한글':3})
            self.assertEqual(read(path),{'한글':3});self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_pending_and_starting_adapters_are_in_backup(self):
        class Store:
            def sync(self,root,scope,files):self.files={p.relative_to(root).as_posix() for p in files}
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(backend='smol',run_name='test',output_root=d,env_root=d,data_dir=d,simulator_repo=d,
                micro_batch=1,accumulate=3,chunk=8,execute_steps=2)
            experiment=FoundationExperiment(cfg);experiment.store=Store()
            start=experiment.run_dir/'candidates/update_001000';pending=experiment.run_dir/'candidates/update_002000'
            for folder in (start,pending):
                folder.mkdir(parents=True);(folder/'adapter.pt').write_bytes(b'model');write(folder/'metadata.json',{'step':1})
            write(experiment.run_dir/'state.json',dict(best=None,pending=dict(folder='candidates/update_002000',checkpoint='candidates/update_001000')))
            experiment.sync()
            self.assertIn('candidates/update_001000/adapter.pt',experiment.store.files)
            self.assertIn('candidates/update_002000/adapter.pt',experiment.store.files)

    def test_training_state_commits_only_after_all_three_scores(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(backend='smol',run_name='state-machine',output_root=d,env_root=d,data_dir=d,simulator_repo=d,
                micro_batch=1,accumulate=3,chunk=8,execute_steps=2,updates=2,eval_interval=1,dev_episodes=1)
            experiment=FoundationExperiment(cfg);experiment.signature=lambda:'fixed';experiment.sync=lambda:None
            write(experiment.run_dir/'state.json',dict(completed=0,best=None,latest=None,pending=None,history=[]))
            write(experiment.run_dir/'runtime_check.json',dict(signature='fixed'))
            def fake_run(command,log,**kwargs):
                job=read(command[-1]);folder=Path(job['destination']);folder.mkdir(parents=True,exist_ok=True)
                write(folder/'metadata.json',dict(step=job['stop']));(folder/'adapter.pt').write_bytes(b'model')
            experiment.run=fake_run
            scores={'easy':.5,'medium':.25,'hard':.1}
            experiment.evaluate=lambda checkpoint,level,seeds,label:{'sort_accuracy':scores[level]}
            experiment.train();state=read(experiment.run_dir/'state.json')
            self.assertEqual(state['completed'],2);self.assertIsNone(state['pending'])
            self.assertAlmostEqual(state['best']['score'],.2*.5+.3*.25+.5*.1)
            self.assertEqual(len(state['history']),2)

    def test_masked_action_padding_is_normalized_to_zero(self):
        import numpy as np,torch
        from foundation_smol import batch_to_torch
        from foundation_octo import training_actions
        stats=dict(proprio_mean=[0.]*26,proprio_std=[1.]*26,action_mean=[.3,-.2,.1,0.],action_std=[.5]*4)
        raw=dict(images=np.zeros((1,2,128,128,3),np.uint8),proprio=np.zeros((1,2,26),np.float32),
            actions=np.zeros((1,2,4,4),np.float32),action_mask=np.zeros((1,2,4),bool),time_mask=np.ones((1,2),bool))
        language=(torch.zeros((1,2),dtype=torch.long),torch.ones((1,2),dtype=torch.bool))
        smol=batch_to_torch(raw,stats,'cpu',language)
        self.assertTrue(torch.equal(smol['action'],torch.zeros_like(smol['action'])))
        octo=training_actions(raw,stats)
        self.assertTrue((octo==0).all())

    @unittest.skipUnless(all((ROOT.parent/l/'trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5').exists() for l in LEVELS),'local RGB data unavailable')
    def test_real_data_split_and_balanced_batch(self):
        import numpy as np
        for level in LEVELS:
            train=set(self.data.ids[level]['train']);valid=set(self.data.ids[level]['valid'])
            self.assertFalse(train & valid);self.assertEqual((len(train),len(valid)),(180,20))
        batch=self.data.batch(LEVELS,np.random.default_rng(9),2,4)
        self.assertEqual(batch['images'].shape,(3,2,128,128,3))
        self.assertEqual(batch['proprio'].shape,(3,2,26))
        self.assertEqual(batch['actions'].shape,(3,2,4,4))
        restored=action_to_env(normalize(batch['actions'],self.data.stats,'action'),self.data.stats)
        self.assertTrue(np.isfinite(restored).all());self.assertTrue((abs(restored)<=1).all())


if __name__=='__main__':unittest.main()
