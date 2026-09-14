"""Joint object-attention training with episode-disjoint validation."""
import json,sys,time
from pathlib import Path
import torch
import shared_stage_train as trainer
from unified_data import load_curriculum
from joint_data import JointWindows
from marso_experiment import save_json


def train(job):
    manifest=json.loads(Path(job['data']).read_text(encoding='utf-8'))
    if manifest.get('mode')!='joint':raise ValueError('Joint dataset manifest required')
    if job.get('warm_start') and not job.get('resume_joint'):
        raise ValueError('New joint experiments must not use an old Easy/Medium/Hard model')
    started=time.monotonic()
    items,train_ids,valid_ids=load_curriculum(manifest['sources'],job['train_config']['seed'])
    def windows(trajectories,ids,history,chunk,training=True,**kwargs):
        selected=set(ids);by_level={k:[i for i in v if i in selected] for k,v in (train_ids if training else valid_ids).items()}
        return JointWindows(trajectories,by_level,history,chunk,training)
    def normalization(trajectories,ids):
        values=torch.cat([trajectories[i]['obs'][:,:26] for i in ids])
        return values.mean(0),values.std(0,unbiased=False).clamp_min(.01)
    names=('load_data','split_data','StageWindows','normalization');old={k:getattr(trainer,k) for k in names}
    trainer.load_data=lambda *args,**kwargs:items
    trainer.split_data=lambda *args:([i for v in train_ids.values() for i in v],[i for v in valid_ids.values() for i in v])
    trainer.StageWindows=windows
    trainer.normalization=normalization
    latest=Path(job['folder'])/'checkpoints/latest.pt'
    start_step=torch.load(latest,map_location='cpu',weights_only=True)['step'] if latest.exists() else 0
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    try:trainer.train(job)
    finally:
        for name,value in old.items():setattr(trainer,name,value)
    elapsed=time.monotonic()-started
    updates=job.get('stop_at',job['train_config']['total_iters'])-start_step
    save_json(Path(job['folder'])/'resource_report.json',dict(elapsed_seconds=elapsed,
        updates_this_call=job.get('stop_at',job['train_config']['total_iters'])-start_step,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None,
        train_episodes={k:len(v) for k,v in train_ids.items()},validation_episodes={k:len(v) for k,v in valid_ids.items()},
        mixing='equal task minibatch counts, at most one sample difference; no failure crops or donor model',
        initialization='resume selected joint model' if job.get('warm_start') else 'random initialization / local joint checkpoint resume'))
    if updates:
        remaining=job['train_config']['total_iters']-job.get('stop_at',job['train_config']['total_iters'])
        print(f'측정 속도: {updates/elapsed:.1f} update/s · 남은 학습 약 {remaining*elapsed/updates/60:.1f}분 (환경 평가·업로드 제외)',flush=True)


if __name__=='__main__':train(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
