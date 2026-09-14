import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ver2/stages'),str(ROOT/'ver2'),str(ROOT)]
import numpy as np
import h5py
from easy_lab import EasyLab,source_bundle
from build_easy_notebook import CONFIG,make_notebook
from stage_train import train
from stage_data import load_data,split_data,StageWindows
from test_stages import state


class EasyLabTests(unittest.TestCase):
    def test_notebook_runs_easy_only_and_keeps_stages_notebook(self):
        nb=make_notebook()
        self.assertEqual(len(nb['cells']),10)
        for i,c in enumerate(nb['cells']):
            code=''.join(c['source']);compile(code,str(i),'exec')
            if i>=4:
                self.assertNotIn('"medium"',code)
                self.assertNotIn('"hard"',code)
        self.assertTrue((ROOT/'notebooks/moveboxes_stages_colab.ipynb').exists())
        self.assertIn('stage_train.py',source_bundle())

    def test_training_and_eval_resume_at_block_boundary_and_preserve_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=dict(CONFIG,profile='smoke',source_run_name=None,warm_start=False,output_root=str(root/'runs'),
                repo_dir=str(root/'repo'),data_dir=str(root/'data'),project_dir=str(ROOT),block_iters=4,max_blocks=2,
                batch_size=4,history=4,chunk_size=16,width=32,heads=4,layers=1,latent_dim=8,validation_batches=1,
                amp=False,ensemble_candidates=[1],test_record_video=False,record_eval_video=False)
            exp=EasyLab(cfg,source_bundle());exp.connect();exp.repo.mkdir()
            path=exp.data/'easy/trajectory.state.pd_ee_delta_pos.physx_cuda.h5';path.parent.mkdir(parents=True)
            with h5py.File(path,'w') as f:
                for i in range(4):
                    g=f.create_group(f'traj_{i}');obs=np.repeat(state()[None],13,axis=0);obs[:,0]=i
                    g.create_dataset('obs',data=obs)
                    action=np.zeros((12,4),np.float32);action[:,3]=1
                    g.create_dataset('actions',data=action)
            folder=exp.run_dir/'easy';folder.mkdir()
            (folder/'inputs_ready.json').write_text('{}')
            original_job=exp.training_job
            def job_without_collection():
                job=original_job();job.pop('recovery_manifest');job['device']='cpu'
                return job
            seen=[];fail=[True]
            def run(command,cwd=None,log=None):
                job=json.loads(Path(command[-1]).read_text())
                if command[1]=='stage_train.py':
                    with patch('stage_train.sync_from_env'):
                        train(job)
                    return
                self.assertEqual(job['level'],'easy')
                self.assertEqual(job['max_steps'],200)
                if 'dev_b02' in job['output'] and fail[0]:
                    fail[0]=False
                    raise RuntimeError('interrupted-evaluation')
                seen.append(job['seeds'])
                score=1. if 'block_02' in job['checkpoint'] else .5
                row=dict(sort_accuracy=score,mean_sorted=2*score,all_placed_rate=float(score==1),mean_steps=120,
                    mis_sort_rate=0,next_pick=dict(stable_grasp_cycles=int(2*score),gripper_sign_reversals=2))
                Path(job['output']).write_text(json.dumps(dict(row,complete=True,n_episodes=len(job['seeds']),
                    episodes=[dict(row,seed=s) for s in job['seeds']],protocol=job)))
            with patch.object(exp,'training_job',side_effect=job_without_collection),patch.object(exp,'run',side_effect=run):
                with self.assertRaisesRegex(RuntimeError,'interrupted-evaluation'):
                    exp.run_blocks()
                self.assertEqual(len(json.loads((folder/'lab_history.json').read_text())['blocks']),1)
                history=exp.run_blocks()
                self.assertEqual(len(history['blocks']),2)
                self.assertEqual(exp._test_candidate('easy')[0].name,'block_02.pt')
                final=exp.final_evaluation()
                self.assertEqual(final['sort_accuracy'],1.)
                self.assertFalse(set(seen[0])&set(seen[-1]))
            self.assertTrue((folder/'checkpoints/block_01.pt').exists())
            self.assertTrue((folder/'checkpoints/block_02.pt').exists())
            with self.assertRaisesRegex(ValueError,'Easy only'):
                exp._test_candidate('hard')

    def test_first_pick_pool_ends_at_first_lift_without_validation_leakage(self):
        import torch
        trajectory=dict(actions=torch.zeros(8,4),stage=torch.tensor([0,0,0,1,1,2,0,0]),
                        gate=torch.zeros(8,dtype=torch.long),source='base')
        other=dict(trajectory,stage=torch.ones(8,dtype=torch.long))
        windows=StageWindows([trajectory,other],[0],4,16,first_pick_fraction=.3)
        self.assertEqual(windows.first_pick,[(0,0),(0,1),(0,2),(0,3)])
