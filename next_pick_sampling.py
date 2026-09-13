"""Increase training exposure to later grasp cycles; never controls the robot.

State index 25 is is_grasped in the pinned WarehouseSort state schema. A later
grasp cycle can be a regrasp of the same parcel, so this is a proxy for next-pick
practice, not a label identifying distinct boxes.
"""
import json
import math
import uuid
from pathlib import Path


def stable_grasp_events(values, min_grasp=3, min_release=3):
    """Return onset indices; require stable release before recognizing another."""
    if min_grasp < 1 or min_release < 1:
        raise ValueError('Stable grasp/release durations must be positive.')
    events = []
    active = False
    high = low = 0
    for t, value in enumerate(values):
        value = float(value)
        if not math.isfinite(value) or min(abs(value), abs(value-1)) > 1e-5:
            raise ValueError('State[25] must be the binary is_grasped signal. Check the dataset schema.')
        if value > 0.5:
            high += 1
            low = 0
            if not active and high >= min_grasp:
                events.append(t-min_grasp+1)
                active = True
        else:
            high = 0
            low += 1
            if low >= min_release:
                active = False
    return events


def sampling_weights(signals, slices, obs_horizon, *, weight=3.0,
                     before=20, after=8, min_grasp=3, min_release=3):
    if not math.isfinite(weight) or not 1 <= weight <= 10:
        raise ValueError('focus_weight must be between 1 and 10.')
    if before < 0 or after < 0 or obs_horizon < 1:
        raise ValueError('Invalid focus window or observation horizon.')
    masks, trajectories = [], []
    for signal in signals:
        events = stable_grasp_events(signal, min_grasp, min_release)
        mask = [False] * len(signal)
        # Retain the first grasp as ordinary training; boost second and later.
        for previous, event in zip(events, events[1:]):
            for t in range(max(previous+min_grasp, event-before), min(len(mask), event+after+1)):
                mask[t] = True
        masks.append(mask)
        trajectories.append(dict(observations=len(signal), stable_grasp_onsets=events,
                                 later_grasp_cycles=max(0, len(events)-1)))
    weights, focused = [], 0
    for trajectory, start, end in slices:
        # The action about to be executed is conditioned on the last observation,
        # not on the beginning of the padded prediction window.
        t = start+obs_horizon-1
        if not 0 <= trajectory < len(masks) or not 0 <= t < len(masks[trajectory]):
            raise ValueError('Dataset sequence indices do not match observations.')
        selected = masks[trajectory][t]
        focused += int(selected)
        weights.append(float(weight if selected else 1))
    if not weights:
        raise ValueError('No training sequences were loaded.')
    audit = dict(sequence_count=len(weights), focused_sequences=focused,
                 focus_fraction_uniform=focused/len(weights),
                 focus_fraction_weighted=focused*weight/sum(weights),
                 trajectories=trajectories,
                 interpretation='Later stable grasp cycles; may include regrasping the same parcel.')
    return weights, audit


def make_sampler(dataset, config_path, seed):
    """Called once by the official trainer after loading demonstration tensors."""
    import torch
    from torch.utils.data import RandomSampler, WeightedRandomSampler

    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    signals = []
    for obs in dataset.trajectories['observations']:
        if obs.ndim != 2 or obs.shape[-1] not in (54, 72, 90):
            raise ValueError(f'Unsupported WarehouseSort state shape: {tuple(obs.shape)}')
        signals.append(obs[:, 25].detach().cpu().tolist())
    effective_weight = cfg['weight'] if cfg['enabled'] else 1.0
    weights, audit = sampling_weights(signals, dataset.slices, dataset.obs_horizon,
        weight=effective_weight, before=cfg['before'], after=cfg['after'],
        min_grasp=cfg['min_grasp'], min_release=cfg['min_release'])
    use_focus = cfg['enabled'] and effective_weight > 1 and audit['focused_sequences'] > 0
    audit.update(config=cfg, sampler='weighted_with_replacement' if use_focus else 'official_uniform')
    destination = config_path.parent/'sampling_audit.json'
    temporary = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    temporary.replace(destination)
    print(f"Next-pick sampling: {audit['sampler']}; focus exposure "
          f"{audit['focus_fraction_uniform']:.1%} -> {audit['focus_fraction_weighted']:.1%}", flush=True)
    if not use_focus:
        print('No weighting applied; see sampling_audit.json for detected grasp cycles.', flush=True)
        return RandomSampler(dataset, replacement=False)
    # Small CPU weights; the original demonstrations and model stay unchanged.
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)


def patched_trainer(source, config_path):
    """Replace exactly one pinned sampler statement, failing on upstream drift."""
    anchor = '    sampler = RandomSampler(dataset, replacement=False)'
    if source.count(anchor) != 1:
        raise ValueError('Official trainer sampler changed; refusing an unverified patch.')
    replacement = ('    from next_pick_sampling import make_sampler\n'
                   f'    sampler = make_sampler(dataset, {str(config_path)!r}, args.seed)')
    result = source.replace(anchor, replacement)
    compile(result, 'train_next_pick.py', 'exec')
    return result
