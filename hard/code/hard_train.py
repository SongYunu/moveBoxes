"""Use the shared resumable trainer with explicit parcel-conditioned supervision."""
import json
import sys
from pathlib import Path
import torch
import shared_stage_train as trainer
from hard_model import StageACT,stage_loss
from hard_data import StageWindows,load_data


def training_forward(model,batch,cfg):
    return model(batch['obs'],batch['previous_stage'],batch['stage'],target=batch['target'])


@torch.no_grad()
def validate(model,data,cfg,device):
    model.eval();generator=torch.Generator().manual_seed(cfg['seed']+991);losses=[]
    for _ in range(cfg['validation_batches']):
        batch=data.batch(cfg['batch_size'],generator,device)
        outputs=model(batch['obs'],batch['previous_stage'],batch['stage'],target=batch['target'],use_target=False)
        loss,_=stage_loss(outputs,batch['actions'],batch['mask'],batch['stage'],batch['gate'],cfg)
        losses.append(float(loss))
    model.train();return sum(losses)/len(losses)


def train(job):
    trainer.StageACT=StageACT;trainer.stage_loss=stage_loss
    trainer.StageWindows=StageWindows;trainer.load_data=load_data
    trainer.training_forward=training_forward;trainer.validate=validate
    return trainer.train(job)


if __name__=='__main__':train(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
