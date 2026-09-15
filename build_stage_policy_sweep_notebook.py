"""Build a no-training Colab that tunes deployment response on the verified anchors."""
import json
from pathlib import Path

from build_stage_anchor_notebook import make_notebook as anchor_notebook


IMPLEMENTATION_COMMIT = '511c935a90738ab2a6de230cd928ae87dba7835d'


def make_notebook():
    nb = anchor_notebook()
    cells = nb['cells'][:5]
    config = ''.join(cells[0]['source'])
    old_ref = next(line.split("'")[3] for line in config.splitlines() if "'project_ref':" in line)
    config = config.replace(old_ref, IMPLEMENTATION_COMMIT)
    config = config.replace("'run_name': 'moveboxes_stage_anchor_v1'",
                            "'run_name': 'moveboxes_stage_policy_sweep_v1'")
    cells[0]['source'] = config.splitlines(keepends=True)

    def add(kind, source, ident):
        cell = dict(cell_type=kind, metadata={'id':ident}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# Medium 34.4% 앵커 · 학습 없는 추론 설정 탐색

Medium 34.4%를 낸 `block_02.pt`의 SHA-256을 그대로 유지합니다. 과거 40.625%도 같은 가중치와 같은 기본 설정에서 다른 8개 시드로 측정된 값이므로 별도 우수 체크포인트가 아닙니다.

집기 직전 XY가 늦게 따라가는 현상에 직접 관련된 **temporal ensemble의 길이와 decay만** 6가지로 비교합니다. 먼저 8개 탐색 시드로 후보를 고른 뒤, 한 번도 선택에 쓰지 않은 16개 시드에서 기본 설정과 일대일 비교합니다. 새 설정이 최소 2개 상자를 더 분류해야 채택합니다. 가중치, stage/gate 임계값, Easy와 Hard는 바꾸지 않습니다.

각 실행은 공식 `eval.py`의 `SORT ACCURACY`만 읽습니다. 중간 결과는 GitHub Release에 백업을 시도하며, 같은 셀을 다시 실행하면 완료된 후보는 재사용합니다. Google Drive는 사용하지 않습니다.
''', 'policy-sweep-guide')

    add('code', '''# 06 · 검증 앵커 복원 + 공식 eval.py 실행기
import hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
from IPython.display import Video, display
from stage_anchor_continue import ANCHORS, package, prepare

MAX_STEPS = 200
SCREEN_SEEDS = list(range(64000, 64008))
CONFIRM_SEEDS = list(range(65000, 65016))
MIN_EXTRA_SORTED = 2
UPSTREAM = Path(CFG['repo_dir'])
OFFICIAL_DEFAULT = UPSTREAM/'conf/eval/default.yaml'
anchor_exp = prepare(experiment, run_suffix='_anchor_policy_sweep_base_v1')
RUN_DIR = Path(anchor_exp.run_dir)

BASELINE_OVERRIDES = dict(ensemble_window=4, temporal_decay=.25)
CANDIDATES = {
    'baseline_w4_d025': BASELINE_OVERRIDES,
    'responsive_w2_d025': dict(ensemble_window=2, temporal_decay=.25),
    'responsive_w2_d075': dict(ensemble_window=2, temporal_decay=.75),
    'responsive_w3_d075': dict(ensemble_window=3, temporal_decay=.75),
    'responsive_w4_d075': dict(ensemble_window=4, temporal_decay=.75),
    'responsive_w4_d150': dict(ensemble_window=4, temporal_decay=1.50),
}

def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def make_candidate(name, overrides):
    kwargs = {} if overrides == BASELINE_OVERRIDES else {'policy_overrides':{'medium':overrides}}
    candidate = package(anchor_exp, folder_name='policy_sweep_candidates/'+name, **kwargs)
    manifest = json.loads((candidate/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['levels']['medium']['checkpoint_sha256'] == ANCHORS['medium']['sha256']
    return candidate

def run_official(candidate, level, label, seeds=None, show_video=False):
    output = RUN_DIR/level/'policy_sweep_official_eval'/label
    output.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        eval_config = OFFICIAL_DEFAULT
    else:
        eval_config = output/'eval_config.yaml'
        eval_config.write_text('eval:\\n  n_episodes: '+str(len(seeds))+\
            '\\n  seeds: '+json.dumps(seeds)+'\\n', encoding='utf-8')
    checkpoint = candidate/'checkpoints'/level/'model.pt'
    sidecar = checkpoint.parent/'policy_config.json'
    identity = dict(level=level, label=label, seeds=seeds,
                    checkpoint_sha256=sha256(checkpoint), policy_sha256=sha256(sidecar))
    result_path = output/'result.json'
    if result_path.is_file():
        saved = json.loads(result_path.read_text(encoding='utf-8'))
        if saved.get('identity') == identity:
            print(f'[재사용] {label}: {saved["score"]:.3%}')
            return saved
    command = [sys.executable, str(UPSTREAM/'eval.py'), 'difficulty='+level,
        'obs_mode=state', 'policy=stage_policy:load_policy',
        'checkpoint='+str(checkpoint), 'eval_config='+str(eval_config),
        'max_episode_steps='+str(MAX_STEPS), 'hydra.run.dir='+str(output)]
    child_env = dict(os.environ)
    child_env['PYTHONPATH'] = str(candidate)+os.pathsep+str(UPSTREAM)+os.pathsep+child_env.get('PYTHONPATH','')
    for key in list(child_env):
        if key.startswith('MOVEBOXES_SYNC_'):
            child_env.pop(key, None)
    log = output/'official_eval.log'
    with log.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen(command, cwd=UPSTREAM, env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                handle.write(line); handle.flush(); print(line, end='', flush=True)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            process.stdout.close()
    text = log.read_text(encoding='utf-8')
    match = re.search(r'SORT ACCURACY:\\s+([0-9.]+)\\s*%', text)
    if code or not match:
        raise RuntimeError(f'official eval failed ({code}); 로그: {log}')
    result = dict(identity=identity, score=float(match.group(1))/100, log=str(log),
                  candidate=str(candidate), policy=json.loads(sidecar.read_text(encoding='utf-8')))
    result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    try:
        anchor_exp.sync_level(level)
    except Exception as error:
        print('GitHub 결과 백업 지연; 로컬 결과는 유지합니다:', error)
    if show_video:
        videos = sorted((output/'videos').rglob('*.mp4'), key=lambda p:p.stat().st_mtime)
        if videos:
            display(Video(str(videos[-1]), embed=True, width=900))
    return result
''', 'policy-sweep-setup')

    add('code', '''# 07 · 8개 탐색 시드 · 6가지 반응성 설정 비교
SCREEN_RESULTS = {}
for name, overrides in CANDIDATES.items():
    candidate = make_candidate(name, overrides)
    result = run_official(candidate, 'medium', 'screen_'+name, SCREEN_SEEDS)
    result['overrides'] = overrides
    SCREEN_RESULTS[name] = result
    print(name, f'{result["score"]:.3%}', overrides)

# 동점이면 원래 설정을 우선하여 불필요한 변경을 막습니다.
SCREEN_WINNER = max(SCREEN_RESULTS,
    key=lambda name:(SCREEN_RESULTS[name]['score'], name == 'baseline_w4_d025'))
screen_summary = {name:row['score'] for name,row in SCREEN_RESULTS.items()}
(RUN_DIR/'medium/policy_sweep_screen.json').write_text(json.dumps(dict(
    seeds=SCREEN_SEEDS, scores=screen_summary, winner=SCREEN_WINNER), indent=2), encoding='utf-8')
print('탐색 승자:', SCREEN_WINNER, CANDIDATES[SCREEN_WINNER],
      f'{SCREEN_RESULTS[SCREEN_WINNER]["score"]:.3%}')
''', 'policy-sweep-screen')

    add('code', '''# 08 · 새 16개 시드에서 baseline과 탐색 승자만 확인
BASELINE_CONFIRM = run_official(
    make_candidate('confirm_baseline', BASELINE_OVERRIDES),
    'medium', 'confirm_baseline', CONFIRM_SEEDS)

if SCREEN_WINNER == 'baseline_w4_d025':
    WINNER_CONFIRM = BASELINE_CONFIRM
    ACCEPT_TUNED_POLICY = False
else:
    winner_overrides = CANDIDATES[SCREEN_WINNER]
    WINNER_CONFIRM = run_official(
        make_candidate('confirm_'+SCREEN_WINNER, winner_overrides),
        'medium', 'confirm_'+SCREEN_WINNER, CONFIRM_SEEDS, show_video=True)
    parcel_trials = len(CONFIRM_SEEDS)*4
    extra_sorted = round((WINNER_CONFIRM['score']-BASELINE_CONFIRM['score'])*parcel_trials)
    ACCEPT_TUNED_POLICY = extra_sorted >= MIN_EXTRA_SORTED
    print('확인 시드 추가 정답 상자:', extra_sorted,
          '· 채택 기준:', MIN_EXTRA_SORTED, '· 채택:', ACCEPT_TUNED_POLICY)

SELECTED_OVERRIDES = CANDIDATES[SCREEN_WINNER] if ACCEPT_TUNED_POLICY else BASELINE_OVERRIDES
confirmation = dict(seeds=CONFIRM_SEEDS, screen_winner=SCREEN_WINNER,
    baseline_score=BASELINE_CONFIRM['score'], winner_score=WINNER_CONFIRM['score'],
    accepted=ACCEPT_TUNED_POLICY, selected_overrides=SELECTED_OVERRIDES)
(RUN_DIR/'medium/policy_sweep_confirmation.json').write_text(
    json.dumps(confirmation, indent=2), encoding='utf-8')
print(json.dumps(confirmation, indent=2))
try:
    anchor_exp.sync_level('medium')
except Exception as error:
    print('GitHub 결과 백업 지연; 로컬 결과는 유지합니다:', error)
''', 'policy-sweep-confirm')

    add('code', '''# 09 · 검증을 통과한 경우에만 설정 적용 · 전 난이도 기본 평가 · ZIP
policy_overrides = {'medium':SELECTED_OVERRIDES} if ACCEPT_TUNED_POLICY else {}
FINAL = package(anchor_exp, policy_overrides=policy_overrides,
                folder_name='final_policy_sweep_candidate')
manifest = json.loads((FINAL/'manifest.json').read_text(encoding='utf-8'))
assert manifest['levels']['medium']['checkpoint_sha256'] == ANCHORS['medium']['sha256']
expected_selection = 'policy_tuned_anchor' if ACCEPT_TUNED_POLICY else 'anchor'
assert manifest['levels']['medium']['selection'] == expected_selection

FINAL_RESULTS = {level:run_official(FINAL, level, 'final_default_'+level,
                                     show_video=(level == 'medium'))
                 for level in ('easy','medium','hard')}
print(json.dumps(dict(accepted=ACCEPT_TUNED_POLICY,
    medium_policy=manifest['levels']['medium']['policy_config'],
    final_default_scores={k:v['score'] for k,v in FINAL_RESULTS.items()}), indent=2))
archive = shutil.make_archive(str(RUN_DIR/'stage_act_validated_policy_submission'),
                              'zip', root_dir=FINAL)
from google.colab import files
print('제출 ZIP:', archive)
files.download(archive)
''', 'policy-sweep-package')

    nb['cells'] = cells
    nb['metadata']['colab']['name'] = 'moveboxes_stage_policy_sweep_colab.ipynb'
    return nb


if __name__ == '__main__':
    notebook = make_notebook()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['metadata'].get('id','cell'), 'exec')
    target = Path(__file__).parent/'notebooks/moveboxes_stage_policy_sweep_colab.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(target)
