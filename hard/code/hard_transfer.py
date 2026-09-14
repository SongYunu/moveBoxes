"""Expand a trained Easy/Medium StageACT checkpoint to the Hard state layout."""
import json
import sys
from pathlib import Path
import torch
from stage_model import StageACT
from stage_data import load_data, split_data, normalization


def feature_pairs(source_dim, destination_dim=90):
    """Return semantically equivalent feature columns for two task sizes."""
    if source_dim not in (54,72) or destination_dim != 90:
        raise ValueError('Only Easy/Medium to Hard transfer is supported')
    sn, dn = (source_dim-36)//9, 6
    pairs = [(i,i) for i in range(26)]
    for parcel in range(sn):
        pairs += [(26+7*parcel+j,26+7*parcel+j) for j in range(7)]
        pairs += [(26+7*sn+2*parcel+j,26+7*dn+2*parcel+j) for j in range(2)]
        pairs += [(source_dim+3*parcel+j,destination_dim+3*parcel+j) for j in range(3)]
        pairs += [(source_dim+3*sn+3*parcel+j,destination_dim+3*dn+3*parcel+j) for j in range(3)]
    pairs += [(26+9*sn+j,26+9*dn+j) for j in range(6)]
    pairs += [(32+9*sn+j,32+9*dn+j) for j in range(4)]
    return pairs


def transplant(source_checkpoint, destination_arch, hard_mean, hard_std, output):
    saved=torch.load(source_checkpoint,map_location='cpu',weights_only=True)
    if saved.get('format')!='moveboxes-stage-act-v1':
        raise ValueError('Donor must be a StageACT v1 checkpoint')
    source_arch=saved['model_config'];source_dim=source_arch['state_dim']
    for key in ('history','chunk_size','width','heads','layers','latent_dim'):
        if source_arch[key]!=destination_arch[key]:
            raise ValueError(f'Donor architecture mismatch: {key}')
    torch.manual_seed(0)
    model=StageACT(destination_arch);state=model.state_dict();donor=saved['model']
    state['obs_mean'].copy_(hard_mean);state['obs_std'].copy_(hard_std)
    pairs=feature_pairs(source_dim);source_features=len(donor['obs_mean']);destination_features=len(state['obs_mean'])
    for src,dst in pairs:
        state['obs_mean'][dst]=donor['obs_mean'][src]
        state['obs_std'][dst]=donor['obs_std'][src]
    state['obs_proj.weight'].zero_();state['obs_proj.bias'].copy_(donor['obs_proj.bias'])
    for src,dst in pairs:state['obs_proj.weight'][:,dst]=donor['obs_proj.weight'][:,src]
    # The posterior is unused by the deployed prior, but transfer it consistently.
    state['posterior.0.weight'].zero_();state['posterior.0.bias'].copy_(donor['posterior.0.bias'])
    history=destination_arch['history']
    for t in range(history):
        for src,dst in pairs:
            state['posterior.0.weight'][:,t*destination_features+dst]=donor['posterior.0.weight'][:,t*source_features+src]
    source_tail=source_features*history;destination_tail=destination_features*history
    state['posterior.0.weight'][:,destination_tail:]=donor['posterior.0.weight'][:,source_tail:]
    special={'obs_mean','obs_std','obs_proj.weight','obs_proj.bias','posterior.0.weight','posterior.0.bias'}
    for key,value in donor.items():
        if key not in special:
            if key not in state or state[key].shape!=value.shape:raise ValueError(f'Unexpected donor tensor: {key}')
            state[key].copy_(value)
    model.load_state_dict(state)
    result=dict(format='moveboxes-stage-act-v1',model_config=destination_arch,model=model.state_dict(),
                step=0,transfer=dict(source_dim=source_dim,source_step=saved.get('step'),mapped_features=len(pairs)))
    Path(output).parent.mkdir(parents=True,exist_ok=True);torch.save(result,output)
    return result['transfer']


def main(job):
    trajectories=load_data(job['data'],job['num_demos'],job['recovery_manifest'])
    train_ids,_=split_data(trajectories,job['seed']);mean,std=normalization(trajectories,train_ids)
    results={}
    for item in job['donors']:
        results[item['level']]=transplant(item['checkpoint'],job['model_config'],mean,std,item['output'])
    Path(job['report']).write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
