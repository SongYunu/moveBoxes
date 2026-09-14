"""Octo Small adaptation in the isolated Python 3.10/JAX environment."""
import copy,json,os,time
from pathlib import Path
import numpy as np
from foundation_common import *


def runtime():
    # TensorFlow is imported by Octo for checkpoint I/O, not GPU training.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)


def to_seven(actions):
    result=np.zeros((*actions.shape[:-1],7),np.float32)
    result[...,:3]=actions[...,:3];result[...,6]=actions[...,3]
    return result


def training_actions(raw,stats):
    normalized=normalize(raw['actions'],stats,'action')
    normalized[~raw['action_mask']]=0.
    return to_seven(normalized)


def make_batch(raw,stats,task):
    from PIL import Image
    images=raw['images'];b,h=images.shape[:2]
    resized=np.stack([np.asarray(Image.fromarray(im).resize((256,256),Image.Resampling.BILINEAR))
        for im in images.reshape(-1,128,128,3)]).reshape(b,h,256,256,3)
    actions=training_actions(raw,stats)
    action_mask=np.zeros_like(actions,dtype=bool)
    action_mask[...,:3]=raw['action_mask'][...,None];action_mask[...,6]=raw['action_mask']
    return dict(observation=dict(image_primary=resized,proprio=normalize(raw['proprio'],stats,'proprio'),
        timestep_pad_mask=raw['time_mask'],pad_mask_dict=dict(image_primary=raw['time_mask'],proprio=raw['time_mask'])),
        task=task,action=actions,action_pad_mask=action_mask)


def trainable_mask(params):
    import flax
    return flax.traverse_util.path_aware_map(lambda path,value:
        not ('hf_model' in '.'.join(path) or 'tokenizers_primary' in '.'.join(path)),params)


def pinned_checkpoint():
    """Make an overlay whose Octo config points at the pinned local T5 snapshot."""
    from huggingface_hub import snapshot_download
    base=Path(snapshot_download('rail-berkeley/octo-small-1.5',revision=OCTO_REV))
    t5=Path(snapshot_download('t5-base',revision=T5_REV))
    overlay=base.parent/f'moveboxes-{OCTO_REV[:12]}-{T5_REV[:12]}'
    overlay.mkdir(exist_ok=True)
    for item in base.iterdir():
        target=overlay/item.name
        if item.name=='config.json' or target.exists() or target.is_symlink():continue
        target.symlink_to(item.resolve(),target_is_directory=item.is_dir())
    config=json.loads((base/'config.json').read_text(encoding='utf-8'))
    config['model']['task_tokenizers']['language']['kwargs']['encoder']=str(t5)
    config['text_processor']['kwargs']['tokenizer_name']=str(t5)
    (overlay/'config.json').write_text(json.dumps(config),encoding='utf-8')
    return str(overlay)


def load(cfg,example,checkpoint=None):
    import flax,jax
    from octo.model.octo_model import OctoModel
    from octo.model.components.tokenizers import LowdimObsTokenizer
    from octo.utils.spec import ModuleSpec
    from octo.utils.train_utils import merge_params
    base=pinned_checkpoint()
    pretrained=OctoModel.load_pretrained(base)
    config=copy.deepcopy(pretrained.config)
    config['model']['observation_tokenizers'].pop('wrist',None)
    config['model']['observation_tokenizers']['proprio']=ModuleSpec.create(LowdimObsTokenizer,
        n_bins=256,bin_type='normal',low=-2.,high=2.,obs_keys=['proprio'])
    # Preserve the pretrained diffusion head (4 steps, 7 channels). Rotation
    # channels 3:6 are masked from loss; gripper stays in channel 6.
    if config['model']['heads']['action']['kwargs']['action_horizon']!=4 or cfg['chunk']!=4:
        raise ValueError('Keep Octo pretrained four-action chunks')
    task=pretrained.create_tasks(texts=[INSTRUCTION]*len(example['images']))
    stats=read(Path(checkpoint)/'metadata.json')['stats'] if checkpoint else example['stats']
    batch=make_batch(example,stats,task)
    model=OctoModel.from_config(config,batch,pretrained.text_processor,rng=jax.random.PRNGKey(cfg['seed']),
        dataset_statistics={})
    new_flat=flax.traverse_util.flatten_dict(model.params);old_flat=flax.traverse_util.flatten_dict(pretrained.params)
    copied=sum(v.size for k,v in new_flat.items() if k in old_flat and v.shape==old_flat[k].shape)
    model=model.replace(params=merge_params(model.params,pretrained.params))
    if copied<20_000_000:raise ValueError('Too few pretrained Octo weights matched')
    del pretrained
    mask=trainable_mask(model.params)
    if checkpoint:
        flat=flax.traverse_util.flatten_dict(model.params);flags=flax.traverse_util.flatten_dict(mask)
        with np.load(Path(checkpoint)/'adapter.npz',allow_pickle=False) as saved:
            if set(saved.files)!={'/'.join(k) for k,v in flags.items() if v}:raise ValueError('Octo adapter keys differ')
            for k,enabled in flags.items():
                if enabled:
                    value=saved['/'.join(k)]
                    if value.shape!=flat[k].shape:raise ValueError('Octo adapter shape differs')
                    flat[k]=jax.numpy.asarray(value)
        model=model.replace(params=flax.traverse_util.unflatten_dict(flat))
    return model,mask,task


def train(job):
    runtime()
    import flax,jax,jax.numpy as jnp,optax
    cfg=job['cfg']
    if cfg.get('device','cuda')=='cuda' and not any(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('Octo JAX CUDA GPU is unavailable')
    data=RGBData(job['data'],cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    checkpoint=job.get('checkpoint');meta=read(Path(checkpoint)/'metadata.json') if checkpoint else None
    if meta and meta['signature']!=job['signature']:raise ValueError('Training config/data/code changed; use a new run_name')
    stats=meta['stats'] if meta else data.stats
    levels=list(LEVELS)*(cfg['micro_batch']//3)
    raw=data.batch(levels,rng,2,4);raw['stats']=stats
    model,mask,task=load(cfg,raw,checkpoint)
    labels=jax.tree_map(lambda x:'train' if x else 'frozen',mask)
    tx=optax.multi_transform({'train':optax.chain(optax.clip_by_global_norm(1.),optax.adamw(cfg['lr'],weight_decay=1e-4)),
        'frozen':optax.set_to_zero()},labels)
    opt_state=tx.init(model.params);key=jax.random.PRNGKey(cfg['seed']);start=meta['step'] if meta else 0
    if meta:rng.bit_generator.state=meta['sample_rng']
    resume=Path(checkpoint)/'resume.msgpack' if checkpoint else None
    if resume and resume.exists():opt_state,key=flax.serialization.from_bytes((opt_state,key),resume.read_bytes())
    def loss_fn(params,batch,key):
        params=jax.tree_map(lambda p,m:p if m else jax.lax.stop_gradient(p),params,mask)
        bound=model.module.bind({'params':params},rngs={'dropout':key})
        embeddings=bound.octo_transformer(batch['observation'],batch['task'],batch['observation']['timestep_pad_mask'],train=True)
        loss,_=bound.heads['action'].loss(embeddings,batch['action'],batch['observation']['timestep_pad_mask'],batch['action_pad_mask'],train=True)
        return loss
    @jax.jit
    def update(params,opt_state,key,batch):
        key,sub=jax.random.split(key);loss,grads=jax.value_and_grad(loss_fn)(params,batch,sub)
        updates,opt_state=tx.update(grads,opt_state,params)
        return optax.apply_updates(params,updates),opt_state,key,loss
    tick=last=time.monotonic();params=model.params
    initial_flat=flax.traverse_util.flatten_dict(params);flat_flags=flax.traverse_util.flatten_dict(mask)
    probe_key=next(k for k,enabled in flat_flags.items() if enabled and 'action' in '/'.join(k) and initial_flat[k].ndim>1)
    probe_before=np.asarray(initial_flat[probe_key]).copy()
    print('Octo Small pretrained / JAX / updates',start,'->',job['stop'],flush=True)
    for step in range(start,job['stop']):
        raw=data.batch(levels,rng,2,4);batch=make_batch(raw,stats,task)
        params,opt_state,key,loss=update(params,opt_state,key,batch);value=float(loss)
        if not np.isfinite(value):raise RuntimeError('Non-finite Octo loss; previous checkpoint retained')
        if time.monotonic()-last>=30 or step+1==job['stop']:
            print(f'{step+1}/{cfg["updates"]} loss={value:.4f} {(step+1-start)/(time.monotonic()-tick):.2f} update/s',flush=True);last=time.monotonic()
    final_flat=flax.traverse_util.flatten_dict(params)
    parameter_delta=float(np.linalg.norm(np.asarray(final_flat[probe_key])-probe_before))
    if job['stop']>start and not parameter_delta>0:raise RuntimeError(f'{probe_key} did not update; checkpoint was not saved')
    validation={};valid_rng=np.random.default_rng(cfg['seed']+1000)
    for level in LEVELS:
        batch=make_batch(data.batch([level]*cfg['micro_batch'],valid_rng,2,4,'valid'),stats,task)
        validation[level]=float(jax.jit(loss_fn)(params,batch,jax.random.PRNGKey(1000)))
    dest=Path(job['destination']);dest.mkdir(parents=True,exist_ok=True)
    flat=final_flat;flags=flat_flags
    with (dest/'adapter.npz.tmp').open('wb') as f:np.savez_compressed(f,**{'/'.join(k):np.asarray(v) for k,v in flat.items() if flags[k]})
    os.replace(dest/'adapter.npz.tmp',dest/'adapter.npz')
    (dest/'resume.msgpack.tmp').write_bytes(flax.serialization.to_bytes((opt_state,key)));os.replace(dest/'resume.msgpack.tmp',dest/'resume.msgpack')
    checkpoint_meta(dest,job,stats,job['stop'],rng)
    memory=jax.devices()[0].memory_stats() or {}
    write(dest/'resource.json',dict(seconds=time.monotonic()-tick,updates=job['stop']-start,
        validation_by_level=validation,parameter_delta_norm=parameter_delta,
        memory={k:int(v) for k,v in memory.items() if isinstance(v,(int,float))}))
    data.close()


class Policy:
    def __init__(self,cfg,checkpoint):
        runtime()
        import jax
        self.cfg=cfg;self.stats=read(Path(checkpoint)/'metadata.json')['stats']
        raw=dict(images=np.zeros((1,2,128,128,3),np.uint8),proprio=np.zeros((1,2,26),np.float32),
            actions=np.zeros((1,2,4,4),np.float32),action_mask=np.ones((1,2,4),bool),time_mask=np.ones((1,2),bool),stats=self.stats)
        self.model,_,self.task=load(cfg,raw,checkpoint);self.reset(0)

    def reset(self,seed):
        import jax
        self.key=jax.random.PRNGKey(seed);self.images=[];self.states=[];self.queue=[]

    def act(self,image,state):
        import jax
        self.images.append(image);self.states.append(state)
        self.images=self.images[-2:];self.states=self.states[-2:]
        if not self.queue:
            images=([self.images[0]] if len(self.images)==1 else [])+self.images
            states=([self.states[0]] if len(self.states)==1 else [])+self.states
            raw=dict(images=np.array(images)[None],proprio=np.array(states)[None],actions=np.zeros((1,2,4,4),np.float32),
                action_mask=np.ones((1,2,4),bool),time_mask=np.array([[len(self.images)>1,True]]))
            batch=make_batch(raw,self.stats,self.task);self.key,key=jax.random.split(self.key)
            actions=np.asarray(self.model.sample_actions(batch['observation'],batch['task'],rng=key))[0]
            four=np.concatenate((actions[:,:3],actions[:,6:7]),axis=-1)
            self.queue=list(action_to_env(four,self.stats)[:self.cfg['execute_steps']])
        return self.queue.pop(0)
