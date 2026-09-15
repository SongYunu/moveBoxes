"""Read-only checkpoint audit and matched-seed A/B/C evaluation. No training/sync.

Run from an installed official starter directory (contains conf/):
  python /path/to/moveBoxes/ver2/stages/stage_compare.py comparison.json
Use --audit-only --device cpu for checkpoint/data checks without ManiSkill.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(Path.cwd())]
import numpy as np
import torch
from stage_chunk_policy import load_chunk_stage
from stage_schema import STAGES

LEVELS = dict(easy=54, medium=72, hard=90)
WEIGHTS = dict(easy=.2, medium=.3, hard=.5)
VARIANTS = ('A', 'B', 'C')
SOURCES = ('stage_compare.py', 'stage_chunk_policy.py', 'stage_policy.py',
           'stage_model.py', 'stage_schema.py', '../act_v2_model.py')


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def fields(dim):
    n = (dim - 36) // 9
    return dict(qpos=[0, 9], qvel=[9, 18], tcp_xyz_quaternion=[18, 25], is_grasped=[25, 26],
                parcel_poses=[26, 26 + 7*n], parcel_tags=[26 + 7*n, 26 + 9*n],
                bin_positions=[26 + 9*n, 32 + 9*n], bin_colors=[32 + 9*n, 36 + 9*n])


def variant_config(base, variant, config):
    # Do not let flags in a previously exported sidecar alter the A baseline.
    options = dict(base)
    options.update(stage_aware_chunk=variant != 'A', gripper_fsm=variant == 'C')
    if variant != 'A':
        options.pop('act_horizon', None)  # Stage-specific execution lengths below.
    options['stage_horizons'] = config.get('stage_horizons', dict(pick=2, carry=6, place=2, done=1))
    options['gripper_margin'] = config.get('gripper_margin', .5)
    options['gripper_confirm_steps'] = config.get('gripper_confirm_steps', 2)
    return options


def load_spec(spec, dim):
    checkpoint = Path(spec['checkpoint']).resolve()
    if spec.get('checkpoint_sha256') and digest(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from the selected baseline')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('format') != 'moveboxes-stage-act-v1':
        raise ValueError('A/B/C supports Stage ACT v1 weights only; do not use Hard target/Object ACT weights')
    if saved['model_config']['state_dim'] != dim:
        raise ValueError('Checkpoint difficulty/state dimension mismatch')
    if 'policy_config' in spec:
        base = dict(spec['policy_config'])
    else:
        sidecar = Path(spec.get('policy_config_path', checkpoint.parent / 'policy_config.json'))
        base = json.loads(sidecar.read_text(encoding='utf-8'))
    if base.get('model_config', saved['model_config']) != saved['model_config']:
        raise ValueError('Policy sidecar architecture mismatch')
    # Require the tuned settings explicitly, rather than silently selecting defaults.
    for key in ('ensemble_window', 'temporal_decay', 'gate_threshold', 'stage_threshold'):
        if key not in base:
            raise ValueError(f'Missing baseline setting: {key}')
    base['model_config'] = saved['model_config']
    return checkpoint, base, dict(format=saved['format'], step=saved.get('step'),
        checkpoint_sha256=digest(checkpoint), model_config=saved['model_config'],
        policy_config=base, state_fields=fields(dim))


@torch.no_grad()
def audit(spec, dim, config, device):
    checkpoint, base, report = load_spec(spec, dim)
    if spec.get('data'):
        import h5py
        from stage_labels import annotate
        counts = np.zeros(4, dtype=np.int64)
        grips = np.zeros((4, 2), dtype=np.int64)
        with h5py.File(spec['data'], 'r') as handle:
            keys = sorted((k for k in handle if k.startswith('traj_')), key=lambda s: int(s.split('_')[-1]))
            for name in keys:
                obs = np.asarray(handle[name]['obs'], dtype=np.float32)
                actions = np.asarray(handle[name]['actions'], dtype=np.float32)
                if obs.shape != (len(actions)+1, dim) or actions.shape != (len(actions), 4):
                    raise ValueError('Dataset observation/action alignment mismatch')
                if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                    raise ValueError('Non-finite demonstration')
                stage = annotate(obs)['stage']
                counts += np.bincount(stage, minlength=4)
                for i in range(4):
                    grips[i] += [int(((stage == i) & (actions[:, 3] < 0)).sum()),
                                 int(((stage == i) & (actions[:, 3] >= 0)).sum())]
            obs = torch.from_numpy(np.asarray(handle[keys[0]]['obs'][:40], dtype=np.float32))
        report.update(trajectories=len(keys), stage_counts=dict(zip(STAGES, counts.tolist())),
            stage_gripper_counts={s: dict(close=int(g[0]), open=int(g[1])) for s, g in zip(STAGES, grips)})
    else:
        obs = torch.zeros(40, dim)
    report['sanity_input'] = 'demonstration replay (not a rollout)' if spec.get('data') else 'synthetic zeros'
    report['variants'] = {}
    for variant in VARIANTS:
        policy = load_chunk_stage(checkpoint, obs[:1], SimpleNamespace(shape=(4,)), device,
                                  **variant_config(base, variant, config))
        first = None
        for frame in obs:
            action = policy.act(frame[None])
            assert action.shape == (1, 4) and torch.isfinite(action).all() and action.abs().max() <= 1
            assert action[0, 3].item() in (-1., 1.) and 0 <= int(policy.stage[0]) < 4
            if first is None:
                first = action.clone()
        decoder_calls = getattr(policy, 'decoder_calls', len(obs))
        policy.reset()
        torch.testing.assert_close(policy.act(obs[:1]), first, rtol=0, atol=0)
        report['variants'][variant] = dict(load_action_reset='passed', frames=len(obs), decoder_calls=decoder_calls)
    if digest(checkpoint) != report['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed during audit')
    return report


class Diagnostics:
    """Monitor public observations and emitted actions; never change the policy."""
    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        self.policy.reset()
        self.grasped, self.signs, self.stages, self.resets = [], [], [], 0

    def act(self, obs, deterministic=True):
        action = self.policy.act(obs, deterministic=deterministic)
        if action.shape != (1, 4) or not torch.isfinite(action).all() or action.abs().max() > 1:
            raise ValueError('Invalid policy action during rollout')
        state = obs['state'] if isinstance(obs, dict) else obs
        self.grasped.append(bool(state[0, 25].item() > .5))
        self.signs.append(int(action[0, 3].item()))
        self.stages.append(int(self.policy.stage[0]))
        decision = self.policy.last_decision
        self.resets += int(decision['accepted'][0] and
            (decision['previous'][0] != decision['stage'][0] or decision['gate'][0] == 2))
        return action

    def report(self):
        # Grasp loss includes release; report only losses observed in CARRY as a proxy.
        return dict(grasp_observed=any(self.grasped),
            possible_carry_drops=sum(a and not b and self.stages[i-1] == 1
                                     for i, (a, b) in enumerate(zip(self.grasped, self.grasped[1:]), 1)),
            gripper_sign_reversals=sum(a != b for a, b in zip(self.signs, self.signs[1:])),
            stage_visits={s: self.stages.count(i) for i, s in enumerate(STAGES)},
            buffer_invalidations=self.resets, policy_calls=len(self.signs),
            decoder_calls=getattr(self.policy, 'decoder_calls', len(self.signs)))


def summarize(results, output):
    table = {}
    for variant in VARIANTS:
        row = {level: results.get(level, {}).get(variant, {}).get('sort_accuracy') for level in LEVELS}
        row['overall'] = sum(WEIGHTS[k]*row[k] for k in LEVELS) if all(row[k] is not None for k in LEVELS) else None
        table[variant] = row
    save(Path(output)/'comparison.json', dict(results=results, table=table,
        note='Matched-seed development comparison; not held-out scores. C is experimental learned-logit filtering.'))
    lines = ['| Variant | Easy | Medium | Hard | Overall |', '|---|---:|---:|---:|---:|']
    for variant, row in table.items():
        cells = ['N/A' if value is None else f'{100*value:.3f}%' for value in row.values()]
        lines.append('| ' + ' | '.join([variant] + cells) + ' |')
    Path(output, 'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return table


def evaluate(config, device='cuda'):
    from warehouse_sort.utils import compose_cfg, make_env, rollout_metrics
    from omegaconf import OmegaConf
    import warehouse_sort.utils as utils
    import warehouse_sort.env as env_module
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime for simulator evaluation')
    seeds = config.get('seeds', list(range(61000, 61008)))
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Use a nonempty list of unique integer evaluation seeds')
    max_steps = config.get('max_steps', 200)
    if type(max_steps) is not int or max_steps < 2:
        raise ValueError('max_steps must be an integer >= 2')
    output = Path(config['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for level, spec in config['levels'].items():
        checkpoint, base, metadata = load_spec(spec, LEVELS[level])
        cfg = compose_cfg(['difficulty='+level, 'obs_mode=state', 'max_episode_steps='+str(max_steps)])
        env, _ = make_env(cfg, 'state', cfg.randomization, num_envs=1)
        try:
            obs, _ = env.reset(seed=0)
            state = obs['state'] if isinstance(obs, dict) else obs
            if tuple(state.shape) != (1, LEVELS[level]):
                raise ValueError('Live environment state shape mismatch')
            results[level] = {}
            for variant in VARIANTS:
                options = variant_config(base, variant, config)
                protocol = dict(level=level, variant=variant, seeds=seeds, max_steps=max_steps,
                    checkpoint_sha256=metadata['checkpoint_sha256'], policy_config=options,
                    environment=OmegaConf.to_container(cfg, resolve=True), torch_version=str(torch.__version__),
                    device=str(device), source_sha256={p: digest(HERE/p) for p in SOURCES},
                    environment_sha256={k: digest(p) for k, p in
                        (('utils', utils.__file__), ('env', env_module.__file__))})
                path = output/level/f'{variant}.json'
                previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                if previous and previous['protocol'] != protocol:
                    raise ValueError(f'Existing result has different inputs/settings: {path}; use a new output directory')
                rows = previous.get('episodes', [])
                if [r['seed'] for r in rows] != seeds[:len(rows)] or len(rows) > len(seeds):
                    raise ValueError('Stored seed prefix mismatch')
                policy = load_chunk_stage(checkpoint, obs, env.single_action_space, device, **options)
                agent = Diagnostics(policy)
                for seed in seeds[len(rows):]:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    # Official rollout_metrics does not reset recurrent policies.
                    # Use one episode per call and explicitly reset all policy state.
                    agent.reset()
                    tick = time.monotonic()
                    with torch.no_grad():
                        metrics = rollout_metrics(env, agent, torch.device(device), 1, [seed], max_steps)
                    rows.append(dict(seed=seed, **metrics, diagnostics=agent.report(),
                        incomplete_at_limit=not bool(metrics['all_placed_rate']),
                        elapsed_seconds=time.monotonic()-tick))
                    score = sum(r['sort_accuracy'] for r in rows)/len(rows)
                    save(path, dict(protocol=protocol, episodes=rows, sort_accuracy=score,
                        complete=len(rows) == len(seeds)))
                    print(f'{level} {variant} seed={seed}: {metrics["sort_accuracy"]:.4f}', flush=True)
                results[level][variant] = dict(sort_accuracy=sum(r['sort_accuracy'] for r in rows)/len(rows),
                    n_episodes=len(rows), checkpoint_sha256=metadata['checkpoint_sha256'], policy_config=options)
                summarize(results, output)
        finally:
            env.close()
        if digest(checkpoint) != metadata['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed during evaluation')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if not config.get('levels') or set(config['levels']) - set(LEVELS):
        raise ValueError('Provide at least one of easy/medium/hard')
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    reports = {level: audit(spec, LEVELS[level], config, args.device) for level, spec in config['levels'].items()}
    save(Path(config['output'])/'audit.json', reports)
    print(json.dumps(reports, indent=2, ensure_ascii=False), flush=True)
    if not args.audit_only:
        evaluate(config, args.device)


if __name__ == '__main__':
    main()
