"""Reuse the resumable small StageACT trainer with canonical multitask windows."""
import json,sys,time
from pathlib import Path
import torch
import shared_stage_train as trainer
from unified_data import load_curriculum,CurriculumWindows
from marso_experiment import save_json


def train(job):
    manifest=json.loads(Path(job['data']).read_text())
    trajectories,train_ids,valid_ids=load_curriculum(manifest['sources'],job['train_config']['seed'])
    active=manifest['active'];arch=job['model_config'];cfg=job['train_config']
    def windows(items,ids,history,chunk,training=True,**kwargs):
        wanted=set(ids);by_level={k:[i for i in v if i in wanted] for k,v in (train_ids if training else valid_ids).items()}
        if not training:by_level={active:by_level[active]}
        return CurriculumWindows(items,by_level,history,chunk,active,manifest.get('focus'),training,
            cfg['replay_fraction'],cfg['speed_bonus'],cfg.get('focus_fraction',.5),cfg.get('contact_sampling',False))
    names=('load_data','split_data','StageWindows');old={k:getattr(trainer,k) for k in names}
    trainer.load_data=lambda *args,**kwargs:trajectories
    trainer.split_data=lambda *args:([i for v in train_ids.values() for i in v],[i for v in valid_ids.values() for i in v])
    trainer.StageWindows=windows
    started=time.monotonic()
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    try:trainer.train(job)
    finally:
        for key,value in old.items():setattr(trainer,key,value)
    elapsed=time.monotonic()-started
    report=dict(elapsed_seconds=elapsed,requested_updates=job['train_config']['total_iters'],
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else None,
        training_episodes={k:len(v) for k,v in train_ids.items()},validation_episodes={k:len(v) for k,v in valid_ids.items()},
        verified_success_episodes={k:sum(trajectories[i]['verified_success'] for i in ids) for k,ids in train_ids.items()},
        crop_kind='expert demonstration windows matched to failed parcel/stage; not on-policy DAgger corrections',
        focus=manifest.get('focus',{}))
    save_json(Path(job['folder'])/'resource_report.json',report)


if __name__=='__main__':train(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
