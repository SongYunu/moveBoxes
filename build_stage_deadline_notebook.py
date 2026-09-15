"""Build the no-Drive, resumable State Stage ACT deadline Colab."""
import json
from pathlib import Path

from build_stage_notebook import CONFIG as STAGE_CONFIG, make_notebook as stage_notebook


CONFIG = dict(
    STAGE_CONFIG,
    run_name='moveboxes_stage_chunk_deadline_v1',
    project_ref='stage-act-chunk-compare',
    output_root='/content/moveboxes_runs',
    data_source='/content/moveboxes_data_cache/marso_state_data.zip',
    download_cache='/content/moveboxes_data_cache',
    save_freq=1000,
    console_interval_seconds=10,
)


def make_notebook(config=None):
    cfg = dict(CONFIG, **(config or {}))
    base = stage_notebook(cfg)
    cells = base['cells'][:5]
    cells[0]['source'][0] = '# 01 · STATE STAGE ACT 단일 정책 CONFIG · GitHub 중단 복구 · Drive 미사용\n'

    def add(kind, text, ident):
        cell = dict(cell_type=kind, metadata={'id': ident}, source=text.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''## 실행 순서

03 셀은 `GH_TOKEN` 환경변수, Colab Secrets, 비공개 입력창 순서로 GitHub 토큰을 받습니다. 같은 `run_name`의 **현재 학습 상태만** GitHub Release에서 복원합니다. 과거 best checkpoint를 자동으로 가져오지 않습니다.

04~05에서 T4 환경과 state dataset을 준비합니다. 각 난이도는 recovery 수집 후 Stage ACT를 학습합니다. 수집은 매 시도, 학습은 매 1,000 iteration마다 GitHub에 동기화됩니다. 런타임이 끊기면 새 런타임에서 01~05와 해당 난이도의 수집·학습 셀을 다시 실행하면 이어집니다.\n\n학습 loss는 action imitation + stage/gate classification + KL을 위한 최적화 신호일 뿐 점수가 아닙니다. 제출 성능은 공식 simulator의 SORT ACCURACY = 올바르게 분류한 parcel 수 / 전체 parcel 수로 평가합니다. 기본 candidate는 loss로 고른 `best_val.pt`가 아니라 현재 학습 진행의 latest.pt를 사용합니다.
''', 'deadline-guide')

    for level in ('easy', 'medium', 'hard'):
        label = level.upper()
        number = len(cells) + 1
        add('code', f'''# {number:02d} · {label} recovery 수집 · 매 시도 GitHub 저장, 재실행 시 이어서 진행
experiment.collect({level!r})
''', f'collect-{level}')
        number = len(cells) + 1
        add('code', f'''# {number:02d} · {label} State Stage ACT 학습 · 로그 표시 + 매 checkpoint 완전 복구 저장
# latest.pt에는 model, optimizer, AMP scaler, sampling/CPU/CUDA RNG가 함께 저장됩니다.
experiment.train({level!r})
''', f'train-{level}')

    add('code', '''# 현재 run의 학습 checkpoint에 단일 stage-aware 실행 정책 적용
import hashlib, importlib.metadata, os, shutil, subprocess, sys
from pathlib import Path
import torch

RUN_DIR = Path(CFG['output_root'])/CFG['run_name']
CANDIDATE = RUN_DIR/'integrated_candidate'
CHECKPOINT_OVERRIDES = {'easy':'', 'medium':'', 'hard':''}
STAGE_HORIZONS = dict(pick=2, carry=6, place=2, done=1)
GRIPPER_MARGIN = .5
GRIPPER_CONFIRM_STEPS = 2
MAX_STEPS = 200
SMOKE_SEED = 61000
DIMS = {'easy':54, 'medium':72, 'hard':90}

# 경로를 따로 지정하지 않으면 loss로 선택한 best_val이 아니라 이 run의 latest를 사용합니다.
selected = {}
for level in DIMS:
    override = CHECKPOINT_OVERRIDES[level]
    if override:
        checkpoint = Path(override).expanduser().resolve()
    else:
        folder = RUN_DIR/level/'checkpoints'
        checkpoint = folder/'latest.pt'\n        if not checkpoint.is_file():\n            checkpoint = None
    if checkpoint is not None:
        selected[level] = checkpoint
if not selected:
    raise FileNotFoundError('먼저 하나 이상의 experiment.train(level) 셀을 실행하세요.')

def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

CANDIDATE.mkdir(parents=True, exist_ok=True)
for source in ('stage_chunk_policy.py','stage_policy.py','stage_model.py','stage_schema.py'):
    shutil.copy2(PROJECT/'ver2/stages'/source, CANDIDATE/source)
shutil.copy2(PROJECT/'ver2/act_v2_model.py', CANDIDATE/'act_v2_model.py')

manifest = dict(policy='state_stage_act_chunk_fsm', project_commit=CFG['project_commit'],
                official_commit=CFG['repo_commit'], run_name=CFG['run_name'], levels={})
level_lines = []
for level, checkpoint in selected.items():
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('format') != 'moveboxes-stage-act-v1':
        raise ValueError(f'{level}: Stage ACT checkpoint가 아닙니다: {saved.get("format")}')
    model_config = saved['model_config']
    if model_config.get('state_dim') != DIMS[level] or model_config.get('chunk_size', 0) < 6:
        raise ValueError(f'{level}: state dimension 또는 trained chunk size 불일치')
    sidecar = checkpoint.parent/'policy_config.json'
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    policy = json.loads(sidecar.read_text(encoding='utf-8'))
    if policy.get('model_config') != model_config:
        raise ValueError(f'{level}: checkpoint/sidecar architecture 불일치')
    policy.update(model_config=model_config, stage_aware_chunk=True,
                  stage_horizons=STAGE_HORIZONS, gripper_fsm=True,
                  gripper_margin=GRIPPER_MARGIN,
                  gripper_confirm_steps=GRIPPER_CONFIRM_STEPS,
                  auto_reset_steps=MAX_STEPS-1)
    policy.pop('act_horizon', None)
    policy.pop('num_inference_steps', None)
    target_dir = CANDIDATE/'checkpoints'/level
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir/'model.pt'
    shutil.copy2(checkpoint, target)
    (target_dir/'policy_config.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    if sha256(target) != sha256(checkpoint):
        raise RuntimeError(f'{level}: checkpoint 사본 해시 불일치')
    manifest['levels'][level] = dict(source=str(checkpoint), checkpoint_sha256=sha256(target),
                                     step=saved.get('step'), model_config=model_config,
                                     policy_config=policy)
    level_lines.append(f'    {level}: {{ checkpoint: checkpoints/{level}/model.pt }}')

submission = 'team: '+json.dumps(CFG['team'])+'\\nstate:\\n  policy: stage_chunk_policy:load_policy\\n  levels:\\n'+'\\n'.join(level_lines)+'\\n'
(CANDIDATE/'submission.yaml').write_text(submission, encoding='utf-8')
(CANDIDATE/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('현재 run checkpoint:', {k:str(v) for k,v in selected.items()})
print('단일 integrated candidate:', CANDIDATE)
''', 'integrated-candidate')

    add('code', '''# 실제 simulator state/action/buffer/reset sanity check
from stage_chunk_policy import load_policy
from warehouse_sort.utils import compose_cfg, make_env

for level in selected:
    cfg = compose_cfg(['difficulty='+level, 'obs_mode=state', 'max_episode_steps='+str(MAX_STEPS)],
                      config_dir=str(Path(CFG['repo_dir'])/'conf'))
    env, _ = make_env(cfg, 'state', cfg.randomization, num_envs=1)
    try:
        obs, _ = env.reset(seed=SMOKE_SEED)
        assert tuple(obs.shape) == (1, DIMS[level])
        agent = load_policy(CANDIDATE/'checkpoints'/level/'model.pt', obs,
                            env.single_action_space, 'cuda')
        with torch.no_grad():
            first = agent.act(obs)
        assert first.shape == (1,4) and torch.isfinite(first).all()
        assert first.abs().max() <= 1 and first[0,3].item() in (-1.,1.)
        assert agent.action_buffer.shape[1] == manifest['levels'][level]['model_config']['chunk_size']
        agent.reset()
        assert not agent.history and agent.action_buffer is None and agent.grip_state is None
        obs2, _ = env.reset(seed=SMOKE_SEED)
        with torch.no_grad():
            repeated = agent.act(obs2)
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        agent.step = MAX_STEPS-1
        agent.history.append(torch.full_like(obs2, 123.))
        agent.action_buffer.fill_(123.)
        with torch.no_grad():
            boundary = agent.act(obs2)
        torch.testing.assert_close(first, boundary, rtol=0, atol=0)
        assert agent.step == 1
        env.step(repeated)
        print(level, 'state/action/buffer/manual+official reset/env.step OK')
    finally:
        env.close()
''', 'integrated-sanity')

    add('code', '''# 공식 eval.py · 1 episode smoke
RESULTS = RUN_DIR
SMOKE_CONFIG = RUN_DIR/'integrated_smoke_eval.yaml'
SMOKE_CONFIG.write_text('eval:\\n  n_episodes: 1\\n  seeds: ['+str(SMOKE_SEED)+']\\n', encoding='utf-8')
UPSTREAM = Path(CFG['repo_dir'])

def run_official(level, eval_config, label):
    output = RUN_DIR/level/'integrated_official_eval'/label
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(UPSTREAM/'eval.py'), 'difficulty='+level,
        'obs_mode=state', 'policy=stage_chunk_policy:load_policy',
        'checkpoint='+str(CANDIDATE/'checkpoints'/level/'model.pt'),
        'eval_config='+str(eval_config), 'max_episode_steps='+str(MAX_STEPS),
        'hydra.run.dir='+str(output)]
    child_env = dict(os.environ)
    child_env['PYTHONPATH'] = str(CANDIDATE)+os.pathsep+str(UPSTREAM)+os.pathsep+child_env.get('PYTHONPATH','')
    log = output/'official_eval.log'
    with log.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen(command, cwd=UPSTREAM, env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors='replace', bufsize=1)
        for line in process.stdout:
            print(line, end='')
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f'official eval failed ({level}); {log} 확인')
    experiment.sync_level(level)
    return log

for level in selected:
    run_official(level, SMOKE_CONFIG, 'smoke')
''', 'official-smoke')

    add('code', '''# 공식 conf/eval/default.yaml 평가 · 자체 metric/승자 선택 없음
OFFICIAL_EVAL_CONFIG = UPSTREAM/'conf/eval/default.yaml'
if not OFFICIAL_EVAL_CONFIG.is_file():
    raise FileNotFoundError(OFFICIAL_EVAL_CONFIG)
for level in selected:
    run_official(level, OFFICIAL_EVAL_CONFIG, 'default')
print('공식 raw logs/videos:', {level:str(RUN_DIR/level/'integrated_official_eval'/'default') for level in selected})
''', 'official-default')

    add('code', '''# 선택 사항 · 공식 eval.py로 100-episode 공개 seed 성능 추정
# 계산식은 공식 evaluator 그대로이며 held-out Kaggle 점수는 아닙니다.
BENCHMARK_EPISODES = int(CFG['benchmark_episodes'])
BENCHMARK_SEEDS = list(range(int(CFG['eval_seed_start']),
                             int(CFG['eval_seed_start'])+BENCHMARK_EPISODES))
BENCHMARK_CONFIG = RUN_DIR/'integrated_public_benchmark.yaml'
BENCHMARK_CONFIG.write_text('eval:\\n  n_episodes: '+str(BENCHMARK_EPISODES)+
    '\\n  seeds: '+json.dumps(BENCHMARK_SEEDS)+'\\n', encoding='utf-8')
for level in selected:
    run_official(level, BENCHMARK_CONFIG, 'public_100ep')
print('100-episode 공식 evaluator raw logs:',
      {level:str(RUN_DIR/level/'integrated_official_eval'/'public_100ep'/'official_eval.log')
       for level in selected})
''', 'official-public-benchmark')
    add('code', '''# 단일 candidate ZIP 생성 및 브라우저 다운로드
check = """import json,sys,torch\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom stage_chunk_policy import load_policy\nroot=Path(sys.argv[1])\nmanifest=json.loads((root/'manifest.json').read_text())\nfor level,row in manifest['levels'].items():\n p=root/'checkpoints'/level/'model.pt'\n agent=load_policy(p,torch.zeros(1,row['model_config']['state_dim']),SimpleNamespace(shape=(4,)),'cpu')\n a=agent.act(torch.zeros(1,row['model_config']['state_dim']))\n assert a.shape==(1,4) and torch.isfinite(a).all() and a.abs().max()<=1\n agent.reset()\n print(level,'candidate import/action/reset OK')\n"""
subprocess.run([sys.executable, '-c', check, str(CANDIDATE)], cwd=CANDIDATE, check=True)
archive = shutil.make_archive(str(CANDIDATE), 'zip', CANDIDATE)
from google.colab import files
print('candidate ZIP:', archive)
files.download(archive)
''', 'candidate-download')

    base['cells'] = cells
    base['metadata']['colab']['name'] = 'moveboxes_stage_deadline_colab.ipynb'
    return base


if __name__ == '__main__':
    notebook = make_notebook()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell.get('id', cell['metadata'].get('id','cell')), 'exec')
    path = Path(__file__).parent/'notebooks/moveboxes_stage_deadline_colab.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(path)
