"""Build one integrated Stage ACT policy workflow using the official evaluator."""
import json
from pathlib import Path


def make_notebook():
    cells = []
    def add(kind, source):
        item = dict(cell_type=kind, id=f'deadline-{len(cells)}', metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    add('markdown', '''# State Stage ACT: 단일 deadline 정책

현재 사용할 State Stage ACT 체크포인트에 다음 실행 로직을 한 번에 적용합니다.

`state → learned stage/gate → stage별 action buffer → transition/recovery 즉시 replan → learned gripper hysteresis`

과거 best를 자동 복원하지 않습니다. 새 성능 수치는 별도 계산하지 않고 고정한 공식 repository의 `eval.py`와 `conf/eval/default.yaml` 출력만 사용합니다. 기존 체크포인트와 sidecar는 읽기만 하며, 새 정책은 Google Drive의 별도 candidate 폴더에 생성됩니다.

Colab에서 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택하고 01부터 실행하세요.
''')

    add('code', '''# 01 · 현재 사용할 Stage ACT checkpoint 경로 입력
PROJECT_REF = 'stage-act-chunk-compare'
USE_DRIVE = True
RUN_NAME = 'state_stage_act_deadline_v1'
TEAM = 'my-team'

# Drive를 쓸 경우 /content/drive/MyDrive/... 절대 경로를 적습니다.
# 비워 둔 난이도는 검사·평가·제출 후보에서 제외됩니다.
CHECKPOINTS = {
    'easy': '',
    'medium': '',
    'hard': '',
}

# True: checkpoint와 같은 폴더의 policy_config.json을 기반으로 사용.
# False: 아래 runtime threshold를 사용. 모델 구조는 항상 checkpoint에서 읽고 검증합니다.
USE_SIDECARS = True
RUNTIME_POLICY = dict(ensemble_window=4, temporal_decay=.25,
                      gate_threshold=.65, stage_threshold=.60)

# 하나의 통합 실행 로직
STAGE_HORIZONS = dict(pick=2, carry=6, place=2, done=1)
GRIPPER_MARGIN = .5
GRIPPER_CONFIRM_STEPS = 2
MAX_STEPS = 200
SMOKE_SEED = 61000
''')

    add('code', '''# 02 · 기존 프로젝트와 동일한 공식 simulator 환경 고정
import hashlib, importlib.metadata, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_ROOT = Path('/content/drive/MyDrive/moveboxes_stage_deadline')
else:
    OUTPUT_ROOT = Path('/content/moveboxes_stage_deadline')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

PROJECT = Path('/content/moveBoxes_stage_deadline')
PROJECT_URL = 'https://github.com/SongYunu/moveBoxes.git'
if not PROJECT.exists():
    subprocess.run(['git','clone','--no-checkout',PROJECT_URL,str(PROJECT)],check=True)
remote = subprocess.check_output(['git','remote','get-url','origin'],cwd=PROJECT,text=True).strip()
if remote != PROJECT_URL:
    raise RuntimeError('Dedicated project directory has a different Git remote.')
subprocess.run(['git','fetch','--depth','1','origin',PROJECT_REF],cwd=PROJECT,check=True)
subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=PROJECT,check=True)
PROJECT_COMMIT = subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
sys.path[:0] = [str(PROJECT/'ver2/stages'),str(PROJECT/'ver2'),str(PROJECT)]

from colab_layout import DEFAULT_CONFIG
UPSTREAM = Path('/content/berlin-marso-hackathon')
if not UPSTREAM.exists():
    subprocess.run(['git','clone',DEFAULT_CONFIG['repo_url'],str(UPSTREAM)],check=True)
    subprocess.run(['git','checkout','--detach',DEFAULT_CONFIG['repo_commit']],cwd=UPSTREAM,check=True)
UPSTREAM_COMMIT = subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip()
if UPSTREAM_COMMIT != DEFAULT_CONFIG['repo_commit']:
    raise RuntimeError('Existing official starter differs; set UPSTREAM to a fresh directory.')

required = {'hydra':'hydra-core','omegaconf':'omegaconf','gymnasium':'gymnasium',
            'transforms3d':'transforms3d','h5py':'h5py'}
install = [package for module,package in required.items() if importlib.util.find_spec(module) is None]
for distribution,version in {'mani-skill':'3.0.1','sapien':'3.0.3'}.items():
    try:
        current = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        current = None
    if current != version:
        install.append(distribution+'=='+version)
if install:
    subprocess.run([sys.executable,'-m','pip','install',*install],check=True)
assert importlib.metadata.version('mani-skill') == '3.0.1'
assert importlib.metadata.version('sapien') == '3.0.3'
subprocess.run([sys.executable,'-m','pip','install','-e',str(UPSTREAM),'--no-deps'],check=True)
sys.path.insert(0,str(UPSTREAM))
os.environ['DISPLAY'] = ''
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch, mani_skill, sapien, warehouse_sort
assert torch.cuda.is_available(), 'T4 GPU runtime required'
print('project / official:',PROJECT_COMMIT,UPSTREAM_COMMIT)
print('GPU:',torch.cuda.get_device_name(0))
print('torch / ManiSkill / SAPIEN:',torch.__version__,getattr(mani_skill,'__version__','unknown'),
      getattr(sapien,'__version__','unknown'))
''')

    add('code', '''# 03 · 원본을 보존한 단일 정책 candidate 생성
DIMS = {'easy':54,'medium':72,'hard':90}
selected = {level:Path(path).expanduser().resolve() for level,path in CHECKPOINTS.items() if path}
if not selected:
    raise ValueError('CHECKPOINTS에 최소 한 난이도의 현재 Stage ACT checkpoint 경로를 입력하세요.')

def sha256(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            sha.update(block)
    return sha.hexdigest()

CANDIDATE = OUTPUT_ROOT/(RUN_NAME+'_candidate')
CANDIDATE.mkdir(parents=True,exist_ok=True)
for source in ('stage_chunk_policy.py','stage_policy.py','stage_model.py','stage_schema.py'):
    shutil.copy2(PROJECT/'ver2/stages'/source,CANDIDATE/source)
shutil.copy2(PROJECT/'ver2/act_v2_model.py',CANDIDATE/'act_v2_model.py')

manifest = dict(policy='state_stage_act_chunk_fsm',project_commit=PROJECT_COMMIT,
                official_commit=UPSTREAM_COMMIT,levels={})
level_lines = []
for level,checkpoint in selected.items():
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{level} checkpoint not found: {checkpoint}')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('format') != 'moveboxes-stage-act-v1':
        raise ValueError(f'{level}: requires moveboxes-stage-act-v1, got {saved.get("format")}')
    model_config = saved.get('model_config',{})
    if model_config.get('state_dim') != DIMS[level] or model_config.get('chunk_size',0) < max(STAGE_HORIZONS.values()):
        raise ValueError(f'{level}: checkpoint shape/chunk does not match this policy')
    sidecar = checkpoint.parent/'policy_config.json'
    if USE_SIDECARS:
        if not sidecar.is_file():
            raise FileNotFoundError(f'{level} sidecar not found: {sidecar}')
        policy = json.loads(sidecar.read_text(encoding='utf-8'))
        if policy.get('model_config',model_config) != model_config:
            raise ValueError(f'{level}: checkpoint/sidecar architecture mismatch')
    else:
        policy = dict(RUNTIME_POLICY)
    for key in ('ensemble_window','temporal_decay','gate_threshold','stage_threshold'):
        if key not in policy:
            raise ValueError(f'{level}: missing runtime setting {key}')
    policy.update(model_config=model_config,stage_aware_chunk=True,
                  stage_horizons=STAGE_HORIZONS,gripper_fsm=True,
                  gripper_margin=GRIPPER_MARGIN,gripper_confirm_steps=GRIPPER_CONFIRM_STEPS,
                  auto_reset_steps=MAX_STEPS-1)
    policy.pop('act_horizon',None)
    policy.pop('num_inference_steps',None)
    folder = CANDIDATE/'checkpoints'/level
    folder.mkdir(parents=True,exist_ok=True)
    target = folder/'model.pt'
    shutil.copy2(checkpoint,target)
    (folder/'policy_config.json').write_text(json.dumps(policy,indent=2),encoding='utf-8')
    digest = sha256(target)
    if digest != sha256(checkpoint):
        raise RuntimeError(f'{level}: copied checkpoint changed')
    manifest['levels'][level] = dict(source=str(checkpoint),checkpoint_sha256=digest,
                                     model_config=model_config,policy_config=policy)
    level_lines.append(f'    {level}: {{ checkpoint: checkpoints/{level}/model.pt }}')

submission = 'team: '+json.dumps(TEAM)+'\\nstate:\\n  policy: stage_chunk_policy:load_policy\\n  levels:\\n'+'\\n'.join(level_lines)+'\\n'
(CANDIDATE/'submission.yaml').write_text(submission,encoding='utf-8')
(CANDIDATE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('candidate:',CANDIDATE)
''')

    add('code', '''# 04 · 실제 state observation/action/buffer/reset sanity check
from types import SimpleNamespace
from stage_chunk_policy import load_policy
from warehouse_sort.utils import compose_cfg,make_env

for level in selected:
    cfg = compose_cfg(['difficulty='+level,'obs_mode=state','max_episode_steps='+str(MAX_STEPS)])
    env,_ = make_env(cfg,'state',cfg.randomization,num_envs=1)
    try:
        obs,_ = env.reset(seed=SMOKE_SEED)
        assert tuple(obs.shape) == (1,DIMS[level]), (level,obs.shape)
        agent = load_policy(CANDIDATE/'checkpoints'/level/'model.pt',obs,env.single_action_space,'cuda')
        with torch.no_grad():
            first = agent.act(obs)
        assert first.shape == (1,4) and torch.isfinite(first).all() and first.abs().max() <= 1
        assert first[0,3].item() in (-1.,1.)
        assert agent.action_buffer.shape[1] == manifest['levels'][level]['model_config']['chunk_size']
        agent.reset()
        assert not agent.history and agent.action_buffer is None and agent.grip_state is None
        obs2,_ = env.reset(seed=SMOKE_SEED)
        with torch.no_grad():
            repeated = agent.act(obs2)
        torch.testing.assert_close(first,repeated,rtol=0,atol=0)
        # Official eval.py runs max_episode_steps-1 actions, resets the env, and
        # does not call agent.reset(). The candidate must clear all runtime state.
        agent.step = MAX_STEPS-1
        agent.history.append(torch.full_like(obs2,123.))
        agent.action_buffer.fill_(123.)
        with torch.no_grad():
            auto_repeated = agent.act(obs2)
        torch.testing.assert_close(first,auto_repeated,rtol=0,atol=0)
        assert agent.step == 1 and len(agent.history) == manifest['levels'][level]['model_config']['history']
        env.step(repeated)
        print(level,'state/action/buffer/manual+official-boundary reset/env.step OK',tuple(obs.shape),first.tolist())
    finally:
        env.close()
''')

    add('code', '''# 05 · 공식 eval.py 실행 helper와 1-episode smoke config
RESULTS = OUTPUT_ROOT/(RUN_NAME+'_official_eval')
RESULTS.mkdir(parents=True,exist_ok=True)
SMOKE_CONFIG = RESULTS/'smoke_eval.yaml'
SMOKE_CONFIG.write_text('eval:\\n  n_episodes: 1\\n  seeds: ['+str(SMOKE_SEED)+']\\n',encoding='utf-8')

def run_official(level,eval_config,label):
    output = RESULTS/label/level
    output.mkdir(parents=True,exist_ok=True)
    checkpoint = CANDIDATE/'checkpoints'/level/'model.pt'
    command = [sys.executable,str(UPSTREAM/'eval.py'),'difficulty='+level,'obs_mode=state',
        'policy=stage_chunk_policy:load_policy','checkpoint='+str(checkpoint),
        'eval_config='+str(eval_config),'max_episode_steps='+str(MAX_STEPS),
        'hydra.run.dir='+str(output)]
    child_env = dict(os.environ)
    child_env['PYTHONPATH'] = str(CANDIDATE)+os.pathsep+str(UPSTREAM)+os.pathsep+child_env.get('PYTHONPATH','')
    log = output/'official_eval.log'
    with log.open('w',encoding='utf-8') as handle:
        process = subprocess.Popen(command,cwd=UPSTREAM,env=child_env,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
        for line in process.stdout:
            print(line,end='')
            handle.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f'official eval failed ({level}); see {log}')
    return log

for level in selected:
    run_official(level,SMOKE_CONFIG,'smoke')
''')

    add('code', '''# 06 · 공식 conf/eval/default.yaml 평가
# 결과 수치는 이 공식 eval.py의 raw log만 사용합니다. 실행마다 공식 video도 생성됩니다.
OFFICIAL_EVAL_CONFIG = UPSTREAM/'conf/eval/default.yaml'
if not OFFICIAL_EVAL_CONFIG.is_file():
    raise FileNotFoundError(OFFICIAL_EVAL_CONFIG)
for level in selected:
    run_official(level,OFFICIAL_EVAL_CONFIG,'default')
print('official logs:',RESULTS/'default')
''')

    add('code', '''# 07 · 검증된 단일 candidate ZIP
# Flat module import와 실제 checkpoint action을 다시 검사한 뒤 압축합니다.
check = """import json,sys,torch\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom stage_chunk_policy import load_policy\nroot=Path(sys.argv[1])\nmanifest=json.loads((root/'manifest.json').read_text())\nfor level,row in manifest['levels'].items():\n p=root/'checkpoints'/level/'model.pt'\n agent=load_policy(p,torch.zeros(1,row['model_config']['state_dim']),SimpleNamespace(shape=(4,)),'cpu')\n a=agent.act(torch.zeros(1,row['model_config']['state_dim']))\n assert a.shape==(1,4) and torch.isfinite(a).all() and a.abs().max()<=1\n agent.reset()\n print(level,'candidate import/action/reset OK')\n"""
subprocess.run([sys.executable,'-c',check,str(CANDIDATE)],cwd=CANDIDATE,check=True)
archive = shutil.make_archive(str(CANDIDATE),'zip',CANDIDATE)
print('candidate ZIP:',archive)
''')

    add('code', '''# 08 · candidate ZIP 다운로드
from google.colab import files
files.download(str(CANDIDATE)+'.zip')
''')

    return dict(nbformat=4,nbformat_minor=5,cells=cells,
        metadata=dict(colab=dict(name='moveboxes_stage_deadline_colab.ipynb'),accelerator='GPU',
            kernelspec=dict(display_name='Python 3',language='python',name='python3')))


if __name__ == '__main__':
    notebook = make_notebook()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']),cell['id'],'exec')
    path = Path(__file__).parent/'notebooks/moveboxes_stage_deadline_colab.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(path)
