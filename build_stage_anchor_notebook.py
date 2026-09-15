"""Build a no-Drive Colab for per-difficulty StageACT anchors and sparse-reward RL."""
import json
from pathlib import Path

from build_stage_deadline_notebook import make_notebook as base_notebook

IMPLEMENTATION_COMMIT = '15da10cc5445d6bfcbb15e47389a0e465822dc40'


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

각 RL iteration은 무작위 실전 환경 16개를 200 step 실행합니다. 보상은 환경이 제공하는 `delta success_count`뿐입니다. 성공 상자가 하나도 없는 rollout은 업데이트를 건너뜁니다. 매 iteration마다 모델·optimizer·난수 상태를 저장하므로 셀 재실행 시 이어집니다.
''', 'success-rl-guide')

    add('code', '''# 09 · Medium 성공 보상 RL → 같은 공식 test
medium_rl = prepare_success_rl(anchor_exp, 'medium', iterations=8, num_envs=16,
                               lr=5e-6, xyz_std=.05)
train_success_rl(medium_rl, 'medium')
MEDIUM_RL = package(medium_rl, use_trained={'medium':True,'hard':False},
                    folder_name='medium_success_rl_candidate')
MEDIUM_RESULT = run_official(MEDIUM_RL, 'medium', 'success_rl_default')
MEDIUM_USE_RL = MEDIUM_RESULT['score'] > BASELINE_RESULTS['medium']['score']
print('Medium 선택:', 'success RL' if MEDIUM_USE_RL else '검증 앵커',
      MEDIUM_RESULT['score'], 'vs', BASELINE_RESULTS['medium']['score'])
''', 'success-rl-medium')

    add('code', '''# 10 · Hard 성공 보상 RL → 같은 공식 test
hard_rl = prepare_success_rl(anchor_exp, 'hard', iterations=10, num_envs=16,
                             lr=5e-6, xyz_std=.06)
train_success_rl(hard_rl, 'hard')
HARD_RL = package(hard_rl, use_trained={'medium':False,'hard':True},
                  folder_name='hard_success_rl_candidate')
HARD_RESULT = run_official(HARD_RL, 'hard', 'success_rl_default')
HARD_USE_RL = HARD_RESULT['score'] > BASELINE_RESULTS['hard']['score']
print('Hard 선택:', 'success RL' if HARD_USE_RL else '검증 앵커',
      HARD_RESULT['score'], 'vs', BASELINE_RESULTS['hard']['score'])
''', 'success-rl-hard')

    add('code', '''# 11 · 공식 점수로 자동 선택한 단일 제출 candidate + 최종 재검증 + ZIP
selected = {}
if globals().get('MEDIUM_USE_RL', False):
    selected['medium'] = Path(MEDIUM_RESULT['checkpoint'])
if globals().get('HARD_USE_RL', False):
    selected['hard'] = Path(HARD_RESULT['checkpoint'])
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

