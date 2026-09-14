import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'hard/code'),str(ROOT/'ver2/stages'),str(ROOT/'ver2'),str(ROOT)]
import torch
import h5py,numpy as np
from hard_model import StageACT,stage_loss,parcel_features
from hard_policy import load_stage
from build_hard_v2_notebook import CONFIG,make_notebook
from hard_v2 import HardV2,source_bundle
from marso_experiment import digest


def state(batch=2,history=4):
    x=torch.zeros(batch,history,90);x[...,18:21]=torch.tensor([0.,0.,.17])
    for j in range(6):
        x[...,26+7*j:29+7*j]=torch.tensor([-.05+j*.02,-.1+j*.04,.03])
        x[...,29+7*j]=1
        x[...,68+2*j+(j%2)]=1
    x[...,80:86]=torch.tensor([0.,-.36,0.,0.,.36,0.]);return x


class HardV2Tests(unittest.TestCase):
    def test_target_changes_actions_and_both_heads_receive_gradients(self):
        cfg=dict(state_dim=90,history=4,chunk_size=8,width=32,heads=4,layers=1,latent_dim=8,target_conditioning=True)
        model=StageACT(cfg);obs=state();stage=torch.zeros(2,dtype=torch.long);target=torch.tensor([0,5])
        outputs=model(obs,stage,stage,target=target)
        self.assertEqual(outputs[0].shape,(2,8,4));self.assertEqual(outputs[4].shape,(2,6))
        actions=torch.zeros(2,8,4);mask=torch.ones(2,8);gates=torch.zeros(2,dtype=torch.long)
        loss,_=stage_loss(outputs,actions,mask,stage,gates,dict(kl_weight=0.,stage_loss_weight=.1,gate_loss_weight=.1,target_loss_weight=.5))
        loss.backward();self.assertGreater(float(model.pointer[-1].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.output[-1].weight.grad.abs().sum()),0)

    def test_checkpoint_loads_without_scripted_geometry(self):
        cfg=dict(state_dim=90,history=4,chunk_size=8,width=32,heads=4,layers=1,latent_dim=8,target_conditioning=True)
        model=StageACT(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';torch.save(dict(format='moveboxes-hard-target-act-v2',model_config=cfg,model=model.state_dict(),step=1),path)
            policy=load_stage(path,{'state':state(1,1)[:,0]},type('A',(),{'shape':(4,)})(),'cpu',ensemble_window=1)
            action=policy.act({'state':state(1,1)[:,0]});self.assertEqual(action.shape,(1,4))
            self.assertIn('target',policy.last_decision)

    def test_notebook_bootstrap_and_sources(self):
        nb=make_notebook();self.assertEqual(len(nb['cells']),11)
        for cell in nb['cells']:compile(''.join(cell['source']),'cell','exec')
        for name,source in source_bundle().items():compile(source,name,'exec')
        scope={};exec(''.join(nb['cells'][0]['source']),scope);scope['CFG'].update(project_dir=str(ROOT),profile='smoke')
        def git(command,**kwargs):return 'https://github.com/SongYunu/moveBoxes.git\n' if 'remote' in command else 'a'*40+'\n'
        previous=list(sys.path)
        try:
            with patch('subprocess.run'),patch('subprocess.check_output',side_effect=git),patch('importlib.reload',side_effect=lambda m:m),contextlib.redirect_stdout(io.StringIO()):
                exec(''.join(nb['cells'][1]['source']),scope)
            self.assertIsInstance(scope['experiment'],HardV2)
            self.assertEqual(scope['experiment'].cfg['total_iters']['hard'],12000)
        finally:sys.path[:]=previous

    def test_two_step_cpu_training_uses_target_batches(self):
        import stage_train as shared
        sys.modules['shared_stage_train']=shared
        import hard_train
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);h5=root/'hard.h5'
            raw=state(1,1)[0,0].numpy()
            with h5py.File(h5,'w') as f:
                for i in range(4):
                    g=f.create_group(f'traj_{i}');obs=np.repeat(raw[None],13,0);obs[:,0]=i
                    action=np.zeros((12,4),np.float32);action[:,3]=1
                    g.create_dataset('obs',data=obs);g.create_dataset('actions',data=action)
            collection=root/'collection';collection.mkdir();entries=[]
            for i in range(2):
                obs=np.repeat(raw[None],13,0).astype(np.float32);actions=np.zeros((12,4),np.float32);actions[:,3]=1
                arrays=dict(obs=obs,actions=actions,previous_stage=np.zeros(12,np.int64),stage=np.zeros(12,np.int64),
                    gate=np.zeros(12,np.int64),target=np.full(12,i,np.int64),perturbed=np.zeros(12,bool))
                p=collection/f'episode_{i}.npz';np.savez_compressed(p,**arrays)
                entries.append(dict(file=p.name,sha256=digest(p),accepted=True))
            manifest=collection/'manifest.json';manifest.write_text(json.dumps(dict(complete=True,episodes=entries)))
            arch=dict(state_dim=90,history=4,chunk_size=8,width=32,heads=4,layers=1,latent_dim=8,target_conditioning=True)
            cfg=dict(seed=1,batch_size=4,lr=1e-4,warmup_steps=1,total_iters=2,save_freq=2,validation_batches=1,
                amp=False,position_noise=0.,console_interval_seconds=999,kl_weight=0.,stage_loss_weight=.1,
                gate_loss_weight=.1,target_loss_weight=.5,action_training_mode='prior')
            job=dict(folder=str(root/'run'),data=str(h5),recovery_manifest=str(manifest),num_demos=None,
                model_config=arch,train_config=cfg,policy_config={},source_sha256='hard-v2',device='cpu')
            with patch.object(shared,'sync_from_env'),contextlib.redirect_stdout(io.StringIO()):hard_train.train(job)
            saved=torch.load(root/'run/checkpoints/latest.pt',weights_only=True)
            self.assertEqual(saved['step'],2);self.assertIn('pointer.2.weight',saved['model'])


if __name__=='__main__':unittest.main()
