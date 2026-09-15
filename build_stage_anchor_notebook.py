"""Build a no-Drive Colab for per-difficulty StageACT anchors and sparse-reward RL."""
import json
from pathlib import Path

from build_stage_deadline_notebook import make_notebook as base_notebook

IMPLEMENTATION_COMMIT = '3ef846e0452a0ae106a3738f72b1754f877b4da2'


def make_notebook():
    nb = base_notebook(dict(run_name='moveboxes_stage_anchor_v1', project_ref=IMPLEMENTATION_COMMIT))
    cells = nb['cells'][:5]

    def add(kind, source, ident):
        cell = dict(cell_type=kind, metadata={'id':ident}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# 난이도별 Stage ACT 앵커 + 성공 보상 RL

Easy 100% `block_03.pt`, Medium 40.625% `block_02.pt`, Hard 4.167% `best_val.pt`를 SHA-256으로 확인해 복원합니다. Easy는 동결합니다. Medium과 Hard는 서로 다른 모델·optimizer·결과 폴더를 사용합니다.

추가 학습은 demonstration action을 정답으로 두는 imitation loss를 사용하지 않습니다. 실제 WarehouseSort 환경의 sparse reward, 즉 `success_count`가 증가할 때의 보상만으로 PPO 업데이트합니다. 기존 앵커의 encoder·stage/gate 판단은 동결하고 action decoder만 작게 업데이트하며, 앵커와의 KL 제약으로 급격한 붕괴를 막습니다.

성능 선택은 loss가 아니라 공식 `eval.py`의 `SORT ACCURACY`로 합니다. RL 결과가 같은 seed의 앵커 점수를 **엄격히 초과할 때만** 채택하며 동점과 하락은 앵커를 유지합니다. Drive는 사용하지 않습니다.
''', 'anchor-guide')

    add('code', '''# 06 · 검증 앵커 복원 / 공식 평가 함수
import json, os, re, shutil, subprocess, sys
from pathlib import Path
from IPython.display import Video, display
from stage_anchor_continue import (ANCHORS, package, prepare, prepare_success_rl,
                                   train_success_rl)

MAX_STEPS = 200
MEDIUM_RL_ITERATIONS = 512 # 양의 정수: 상한 없음. 늘린 뒤 재실행하면 이어서 학습
HARD_RL_ITERATIONS = 64    # 양의 정수: 상한 없음
RL_EVAL_EVERY = 32
UPSTREAM = Path(CFG['repo_dir'])
OFFICIAL = UPSTREAM/'conf/eval/default.yaml'
anchor_exp = prepare(experiment, run_suffix='_anchor_success_base_v1')
RUN_DIR = Path(anchor_exp.run_dir)
BASELINE = package(anchor_exp, folder_name='anchor_candidate')

def run_official(candidate, level, label):
    output = RUN_DIR/level/'success_rl_official_eval'/candidate.name/label
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(UPSTREAM/'eval.py'), 'difficulty='+level,
        'obs_mode=state', 'policy=stage_policy:load_policy',
        'checkpoint='+str(candidate/'checkpoints'/level/'model.pt'),
        'eval_config='+str(OFFICIAL), 'max_episode_steps='+str(MAX_STEPS),
        'hydra.run.dir='+str(output)]
    child_env = dict(os.environ)
    child_env['PYTHONPATH'] = str(candidate)+os.pathsep+str(UPSTREAM)+os.pathsep+child_env.get('PYTHONPATH','')
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
    if code:
        raise RuntimeError(f'official eval failed ({level}); {log} 확인')
    match = re.search(r'SORT ACCURACY:\\s+([0-9.]+)\\s*%', log.read_text(encoding='utf-8'))
    if not match:
        raise ValueError(f'공식 점수를 읽지 못했습니다: {log}')
    videos = sorted((output/'videos').rglob('*.mp4'), key=lambda p:p.stat().st_mtime)
    if videos:
        display(Video(str(videos[-1]), embed=True, width=900))
    return {'level':level, 'score':float(match.group(1))/100, 'log':str(log),
            'checkpoint':str(candidate/'checkpoints'/level/'model.pt')}
''', 'anchor-prepare')

    add('code', '''# 07 · 학습 전 공식 default test + 영상
BASELINE_RESULTS = {level:run_official(BASELINE, level, 'anchor_default')
                    for level in ('easy','medium','hard')}
print(json.dumps(BASELINE_RESULTS, indent=2))
''', 'anchor-baseline-test')

    add('markdown', '''## 성공 보상 RL

각 RL iteration은 무작위 실전 환경 16개를 200 step 실행합니다. 보상은 환경이 제공하는 `delta success_count`뿐입니다. 성공 상자가 하나도 없는 rollout은 업데이트를 건너뜁니다. 반복 수에는 상한이 없으며 06번 셀의 `MEDIUM_RL_ITERATIONS`와 `HARD_RL_ITERATIONS`에서 정합니다. `RL_EVAL_EVERY`마다 공식 평가하고 중간 최고 체크포인트를 별도로 저장하므로 뒤의 라운드에서 성능이 떨어져도 좋은 모델을 잃지 않습니다. 반복 수를 더 큰 값으로 바꾸고 재실행하면 기존 optimizer와 체크포인트에서 계속됩니다.
''', 'success-rl-guide')

    add('code', '''# 09 · 8회 단위 공식 평가와 중간 최고 체크포인트 보존
def evaluate_success_round(rl_exp, level, stop, best):
    boundary = rl_exp.run_dir/level/'checkpoints'/f'iteration_{stop:04d}.pt'
    if not boundary.is_file():
        raise FileNotFoundError(f'라운드 체크포인트가 없습니다: {boundary}')
    candidate = package(rl_exp, checkpoint_overrides={level:boundary},
                        folder_name=f'{level}_success_rl_round_{stop:02d}')
    result = run_official(candidate, level, f'success_rl_round_{stop:02d}')
    result.update(iteration=stop, source='success_rl')
    history_path = rl_exp.run_dir/level/'official_rounds.json'
    history = json.loads(history_path.read_text()) if history_path.is_file() else []
    history = [item for item in history if item.get('iteration') != stop]
    history.append({k:v for k,v in result.items() if k != 'checkpoint'})
    history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    if result['score'] > best['score']:
        promoted = rl_exp.run_dir/level/'checkpoints'/'official_best.pt'
        shutil.copy2(result['checkpoint'], promoted)
        result['checkpoint'] = str(promoted)
        best = result
        print(f'{level} {stop}회: 새 최고 공식 점수 {result["score"]:.3%}')
    else:
        print(f'{level} {stop}회: {result["score"]:.3%}; 현재 최고 {best["score"]:.3%} 유지')
    try:
        rl_exp.sync_level(level)
    except Exception as error:
        print('GitHub 백업 지연; 로컬 최고 체크포인트는 유지합니다:', error)
    return best

def evaluation_stops(total, every=RL_EVAL_EVERY):
    if type(total) is not int or total < 1 or type(every) is not int or every < 1:
        raise ValueError('RL 반복 수와 평가 간격은 양의 정수여야 합니다.')
    stops = list(range(every, total+1, every))
    if not stops or stops[-1] != total:
        stops.append(total)
    return stops

# Medium: 설정한 횟수까지 하나의 학습 궤적을 이어서 평가
medium_rl = prepare_success_rl(anchor_exp, 'medium', iterations=MEDIUM_RL_ITERATIONS,
                               num_envs=16, lr=5e-6, xyz_std=.05)
MEDIUM_BEST = dict(BASELINE_RESULTS['medium'], iteration=0, source='anchor')
for stop in evaluation_stops(MEDIUM_RL_ITERATIONS):
    train_success_rl(medium_rl, 'medium', until_iteration=stop)
    MEDIUM_BEST = evaluate_success_round(medium_rl, 'medium', stop, MEDIUM_BEST)
MEDIUM_USE_RL = MEDIUM_BEST['source'] == 'success_rl'
print('Medium 최종 선택:', MEDIUM_BEST)
''', 'success-rl-medium')

    add('code', '''# 10 · Hard도 설정한 횟수까지 이어서 평가
hard_rl = prepare_success_rl(anchor_exp, 'hard', iterations=HARD_RL_ITERATIONS,
                             num_envs=16, lr=5e-6, xyz_std=.06)
HARD_BEST = dict(BASELINE_RESULTS['hard'], iteration=0, source='anchor')
for stop in evaluation_stops(HARD_RL_ITERATIONS):
    train_success_rl(hard_rl, 'hard', until_iteration=stop)
    HARD_BEST = evaluate_success_round(hard_rl, 'hard', stop, HARD_BEST)
HARD_USE_RL = HARD_BEST['source'] == 'success_rl'
print('Hard 최종 선택:', HARD_BEST)
''', 'success-rl-hard')

    add('code', '''# 11 · 공식 점수로 자동 선택한 단일 제출 candidate + 최종 재검증 + ZIP
selected = {}
if globals().get('MEDIUM_USE_RL', False):
    selected['medium'] = Path(MEDIUM_BEST['checkpoint'])
if globals().get('HARD_USE_RL', False):
    selected['hard'] = Path(HARD_BEST['checkpoint'])
FINAL = package(anchor_exp, checkpoint_overrides=selected, folder_name='final_success_rl_candidate')
FINAL_RESULTS = {level:run_official(FINAL, level, 'final_default')
                 for level in ('easy','medium','hard')}
print(json.dumps({'selected':{k:str(v) for k,v in selected.items()},
                  'official_results':FINAL_RESULTS}, indent=2))
archive = shutil.make_archive(str(RUN_DIR/'stage_act_success_rl_submission'), 'zip', root_dir=FINAL)
from google.colab import files
files.download(archive)
''', 'success-rl-package')

    nb['cells'] = cells
    nb['metadata']['colab']['name'] = 'moveboxes_stage_anchor_continue_colab.ipynb'
    return nb


if __name__ == '__main__':
    notebook = make_notebook()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['metadata'].get('id','cell'), 'exec')
    target = Path(__file__).parent/'notebooks/moveboxes_stage_anchor_continue_colab.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(target)
