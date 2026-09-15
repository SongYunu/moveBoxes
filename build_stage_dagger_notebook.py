"""Build the no-Drive Colab for supervised on-policy DAgger correction."""
import json
from pathlib import Path

from build_stage_anchor_notebook import make_notebook as anchor_notebook


IMPLEMENTATION_COMMIT = 'f7af1f95295b5ecb838d7869f1d7af201e4aec5f'


def make_notebook():
    nb = anchor_notebook()
    cells = nb['cells'][:5]
    config = ''.join(cells[0]['source'])
    old_ref = next(line.split("'")[3] for line in config.splitlines() if "'project_ref':" in line)
    config = config.replace(old_ref, IMPLEMENTATION_COMMIT)
    config = config.replace("'run_name': 'moveboxes_stage_anchor_v1'",
                            "'run_name': 'moveboxes_stage_dagger_v1'")
    cells[0]['source'] = config.splitlines(keepends=True)

    def add(kind, source, ident):
        cell = dict(cell_type=kind, metadata={'id':ident}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# Medium Stage ACT · 실제 실패 상태 DAgger 교정

PPO와 sparse reward를 사용하지 않습니다. 34.4% Medium 앵커가 simulator에서 실제로 방문한 상태마다 수집 전용 expert의 교정 행동을 저장하고, 그 데이터를 원본 200개 시연 및 기존 성공 recovery와 함께 supervised 학습합니다.

기존 수집은 expert 행동에 작은 노이즈를 넣은 상태만 보았고 실패 rollout은 버렸습니다. 이 노트북은 learned policy가 만든 XY/Z 오차 상태와 실패 직전 상태도 버리지 않습니다. 각 교정 라벨 다음에 실제 실행 행동이 다르면 action chunk를 1 step에서 끊어, 존재하지 않는 미래 정답을 학습하지 않습니다.

앵커의 observation encoder는 동결합니다. action decoder, stage/gate supervisor와 head만 낮은 learning rate로 업데이트합니다. 매 round는 24개 rollout 수집 → 2,000 update → 공식 `eval.py` 8-seed 평가 순서입니다. 새 16개 확인 시드에서 baseline보다 최소 2개 상자를 더 분류해야 최종 제출에 반영합니다. 실패하면 원래 앵커를 유지합니다. 제출물에는 expert 코드나 규칙 기반 제어가 포함되지 않습니다.
''', 'dagger-guide')

    add('code', '''# 06 · 앵커/DAgger 준비 + 공식 eval.py 실행기
import hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
from IPython.display import Video, display
from stage_anchor_continue import ANCHORS, package, prepare
from stage_dagger import prepare_dagger, run_dagger_round

LEVEL = 'medium'
DAGGER_ROUNDS = 4
EPISODES_PER_ROUND = 24
TRAIN_ITERS_PER_ROUND = 2000
BETAS = (.50, .30, .15, 0.)
SELECTION_SEEDS = list(range(68000, 68008))
CONFIRM_SEEDS = list(range(69000, 69016))
MIN_EXTRA_SORTED = 2
MAX_STEPS = 200

UPSTREAM = Path(CFG['repo_dir'])
OFFICIAL_DEFAULT = UPSTREAM/'conf/eval/default.yaml'
anchor_exp = prepare(experiment, run_suffix='_anchor_dagger_base_v1')
RUN_DIR = Path(anchor_exp.run_dir)
DAGGER_ROOT = prepare_dagger(anchor_exp, LEVEL, run_suffix='dagger_medium_v1',
    rounds=DAGGER_ROUNDS, episodes_per_round=EPISODES_PER_ROUND,
    train_iters=TRAIN_ITERS_PER_ROUND, lr=1e-5, betas=BETAS)

def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def candidate(name, checkpoint=None):
    overrides = {LEVEL:Path(checkpoint)} if checkpoint is not None else None
    result = package(anchor_exp, checkpoint_overrides=overrides,
                     folder_name='dagger_candidates/'+name)
    manifest = json.loads((result/'manifest.json').read_text(encoding='utf-8'))
    if checkpoint is None:
        assert manifest['levels'][LEVEL]['checkpoint_sha256'] == ANCHORS[LEVEL]['sha256']
    return result

def run_official(item, level, label, seeds=None, show_video=False):
    output = RUN_DIR/level/'dagger_official_eval'/label
    output.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        eval_config = OFFICIAL_DEFAULT
    else:
        eval_config = output/'eval_config.yaml'
        eval_config.write_text('eval:\\n  n_episodes: '+str(len(seeds))+\
            '\\n  seeds: '+json.dumps(seeds)+'\\n', encoding='utf-8')
    checkpoint = item/'checkpoints'/level/'model.pt'
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
    child_env['PYTHONPATH'] = str(item)+os.pathsep+str(UPSTREAM)+os.pathsep+child_env.get('PYTHONPATH','')
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
    result = dict(identity=identity, score=float(match.group(1))/100,
                  log=str(log), candidate=str(item))
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
''', 'dagger-setup')

    add('code', '''# 07 · 원본 Medium 앵커를 새 선택 시드에서 먼저 측정
ANCHOR_CHECKPOINT = RUN_DIR/LEVEL/'anchor.pt'
BASELINE_CANDIDATE = candidate('baseline_anchor')
BASELINE_SELECTION = run_official(
    BASELINE_CANDIDATE, LEVEL, 'selection_baseline', SELECTION_SEEDS)
print('Medium 선택 시드 baseline:', f'{BASELINE_SELECTION["score"]:.3%}',
      '· 기존 34.4%는 seeds 63000..63007의 별도 측정값')
''', 'dagger-baseline')

    add('code', '''# 08 · 4개 DAgger round · 수집/학습 중단 시 같은 셀로 재개
BEST_CHECKPOINT = ANCHOR_CHECKPOINT
BEST_SCORE = BASELINE_SELECTION['score']
BEST_ROUND = 0
ROUND_RESULTS = []

for round_number in range(1, DAGGER_ROUNDS+1):
    trained = run_dagger_round(anchor_exp, LEVEL, DAGGER_ROOT,
                               round_number, BEST_CHECKPOINT)
    item = candidate(f'round_{round_number:02d}', trained)
    result = run_official(item, LEVEL, f'selection_round_{round_number:02d}',
                          SELECTION_SEEDS, show_video=True)
    result.update(round=round_number, checkpoint=str(trained))
    ROUND_RESULTS.append(result)
    if result['score'] > BEST_SCORE:
        BEST_CHECKPOINT, BEST_SCORE, BEST_ROUND = trained, result['score'], round_number
        print(f'Round {round_number}: 새 공식 최고 {BEST_SCORE:.3%}')
    else:
        print(f'Round {round_number}: {result["score"]:.3%}; '
              f'현재 최고 {BEST_SCORE:.3%} 유지')
    summary = dict(baseline=BASELINE_SELECTION['score'], best_score=BEST_SCORE,
        best_round=BEST_ROUND, results=[dict(round=row['round'], score=row['score'],
        checkpoint=row['checkpoint']) for row in ROUND_RESULTS])
    (DAGGER_ROOT/'official_rounds.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    try:
        anchor_exp.sync_level(LEVEL)
    except Exception as error:
        print('GitHub round 백업 지연; 로컬 checkpoint는 유지합니다:', error)

print('DAgger 선택 최고:', dict(round=BEST_ROUND, score=BEST_SCORE,
                                  checkpoint=str(BEST_CHECKPOINT)))
''', 'dagger-rounds')

    add('code', '''# 09 · 선택에 쓰지 않은 16개 시드에서 baseline과 최고 DAgger 확인
BASELINE_CONFIRM = run_official(
    candidate('confirm_baseline'), LEVEL, 'confirm_baseline', CONFIRM_SEEDS)

if BEST_ROUND == 0:
    DAGGER_CONFIRM = BASELINE_CONFIRM
    EXTRA_SORTED = 0
    ACCEPT_DAGGER = False
else:
    DAGGER_CONFIRM = run_official(
        candidate('confirm_dagger', BEST_CHECKPOINT), LEVEL,
        'confirm_dagger', CONFIRM_SEEDS, show_video=True)
    EXTRA_SORTED = round((DAGGER_CONFIRM['score']-BASELINE_CONFIRM['score'])*len(CONFIRM_SEEDS)*4)
    ACCEPT_DAGGER = EXTRA_SORTED >= MIN_EXTRA_SORTED

confirmation = dict(seeds=CONFIRM_SEEDS, best_round=BEST_ROUND,
    baseline_score=BASELINE_CONFIRM['score'], dagger_score=DAGGER_CONFIRM['score'],
    extra_sorted=EXTRA_SORTED, required_extra_sorted=MIN_EXTRA_SORTED,
    accepted=ACCEPT_DAGGER)
(DAGGER_ROOT/'confirmation.json').write_text(json.dumps(confirmation, indent=2), encoding='utf-8')
print(json.dumps(confirmation, indent=2))
''', 'dagger-confirm')

    add('code', '''# 10 · 확인을 통과한 학습 정책만 제출 반영 · 전체 난이도 평가 · ZIP
overrides = {LEVEL:BEST_CHECKPOINT} if ACCEPT_DAGGER else None
FINAL = package(anchor_exp, checkpoint_overrides=overrides,
                folder_name='final_validated_dagger_candidate')
manifest_path = FINAL/'manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
if ACCEPT_DAGGER:
    manifest['levels'][LEVEL]['selection'] = 'supervised_dagger'
    manifest['levels'][LEVEL]['dagger_round'] = BEST_ROUND
else:
    assert manifest['levels'][LEVEL]['checkpoint_sha256'] == ANCHORS[LEVEL]['sha256']
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

FINAL_RESULTS = {level:run_official(FINAL, level, 'final_default_'+level,
                                     show_video=(level == LEVEL))
                 for level in ('easy','medium','hard')}
print(json.dumps(dict(dagger_accepted=ACCEPT_DAGGER, best_round=BEST_ROUND,
    final_default_scores={key:value['score'] for key,value in FINAL_RESULTS.items()}), indent=2))
archive = shutil.make_archive(str(RUN_DIR/'stage_act_validated_dagger_submission'),
                              'zip', root_dir=FINAL)
from google.colab import files
print('제출 ZIP:', archive)
files.download(archive)
''', 'dagger-package')

    nb['cells'] = cells
    nb['metadata']['colab']['name'] = 'moveboxes_stage_dagger_colab.ipynb'
    return nb


if __name__ == '__main__':
    notebook = make_notebook()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['metadata'].get('id','cell'), 'exec')
    target = Path(__file__).parent/'notebooks/moveboxes_stage_dagger_colab.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(target)
