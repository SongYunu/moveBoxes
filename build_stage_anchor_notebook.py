"""Build one no-Drive Colab for per-difficulty StageACT anchors and continuation."""
import json
from pathlib import Path

from build_stage_deadline_notebook import make_notebook as base_notebook


def make_notebook():
    nb = base_notebook(dict(run_name='moveboxes_stage_anchor_v1',
                            project_ref='stage-act-chunk-compare'))
    cells = nb['cells'][:5]

    def add(kind, source, ident):
        cell = dict(cell_type=kind, metadata={'id':ident},
                    source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# 난이도별 검증 Stage ACT 복원 및 독립 추가 학습

Easy 100% `block_03.pt`, Medium 40.625% `block_02.pt`, Hard 4.167% `best_val.pt`를 SHA-256으로 확인해 각각 복원합니다. Easy는 고정하며 Medium과 Hard는 서로 다른 초기값·optimizer·결과 폴더로 학습합니다. 한 난이도의 학습은 다른 난이도 가중치를 변경하지 않습니다.

01~05는 기존 GitHub 토큰, 데이터 다운로드, T4 설치 과정과 같습니다. Drive는 사용하지 않습니다. 06은 세 앵커와 각 난이도의 기존 recovery 시연을 복원합니다. 07은 학습 전 공식 테스트입니다. 09와 11은 Medium/Hard 추가 학습입니다. 500 step마다 optimizer·AMP scaler·RNG까지 GitHub Release에 저장합니다.

추가 학습 모델은 원본 앵커를 덮어쓰지 않습니다. 마지막 셀에서 공식 결과를 확인한 뒤 `USE_TRAINED`를 직접 지정합니다. 기본값은 모두 `False`라서 검증 앵커가 유지됩니다.
''', 'anchor-guide')

    add('code', '''# 06 · 세 앵커 + 기존 recovery 시연 복원 / baseline candidate 생성
import json, os, shutil, signal, subprocess, sys
from pathlib import Path
from IPython.display import Video, display
from stage_anchor_continue import ANCHORS, prepare, train, package

CONTINUE_ITERS = {'medium':8000, 'hard':12000}
CONTINUE_LR = 2e-5
MAX_STEPS = 200
UPSTREAM = Path(CFG['repo_dir'])
OFFICIAL = UPSTREAM/'conf/eval/default.yaml'
anchor_exp = prepare(experiment, iterations=CONTINUE_ITERS, lr=CONTINUE_LR)
RUN_DIR = Path(anchor_exp.run_dir)
BASELINE = package(anchor_exp, use_trained={'medium':False,'hard':False},
                   folder_name='anchor_candidate')
print('학습 전 candidate:', BASELINE)
for level, spec in ANCHORS.items():
    print(level, spec)

def run_official(candidate, level, label):
    output = RUN_DIR/level/'anchor_official_eval'/candidate.name/label
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
    videos = sorted((output/'videos').rglob('*.mp4'), key=lambda p:p.stat().st_mtime)
    if videos:
        display(Video(str(videos[-1]), embed=True, width=900))
    return log
''', 'anchor-prepare')

    add('code', '''# 07 · 학습 전 공식 default test · 난이도마다 등록된 앵커 사용
for level in ('easy','medium','hard'):
    run_official(BASELINE, level, 'anchor_default')
''', 'anchor-baseline-test')

    add('markdown', '''## 추가 학습

Easy는 100/100 모델을 그대로 둡니다. Medium과 Hard만 각자의 앵커에서 시작합니다. 셀 재실행 시 같은 난이도의 `latest.pt`와 optimizer/RNG가 복구됩니다. 성능 판단에는 공식 `eval.py` 출력만 사용합니다.
''', 'anchor-train-guide')

    add('code', '''# 09 · Medium block_02.pt에서 독립 추가 학습 → 공식 test + 영상
train(anchor_exp, 'medium')
MEDIUM_TRAINED = package(anchor_exp, use_trained={'medium':True,'hard':False},
                         folder_name='medium_trained_candidate')
run_official(MEDIUM_TRAINED, 'medium', 'trained_default')
''', 'anchor-train-medium')

    add('code', '''# 10 · Medium 앵커를 다시 공식 test하여 같은 실행의 직접 비교 자료 생성
run_official(BASELINE, 'medium', 'anchor_repeat')
''', 'anchor-medium-reference')

    add('code', '''# 11 · Hard best_val.pt에서 독립 추가 학습 → 공식 test + 영상
train(anchor_exp, 'hard')
HARD_TRAINED = package(anchor_exp, use_trained={'medium':False,'hard':True},
                       folder_name='hard_trained_candidate')
run_official(HARD_TRAINED, 'hard', 'trained_default')
''', 'anchor-train-hard')

    add('code', '''# 12 · Hard 앵커를 다시 공식 test하여 같은 실행의 직접 비교 자료 생성
run_official(BASELINE, 'hard', 'anchor_repeat')
''', 'anchor-hard-reference')

    add('code', '''# 13 · 최종 난이도별 선택 및 제출 ZIP
# 위 공식 test에서 추가 학습 결과가 더 좋을 때만 해당 값을 True로 바꾸세요.
USE_TRAINED = {'medium':False, 'hard':False}
FINAL = package(anchor_exp, use_trained=USE_TRAINED, folder_name='final_candidate')
print(json.dumps(json.loads((FINAL/'manifest.json').read_text()), indent=2))
archive = shutil.make_archive(str(RUN_DIR/'stage_act_per_difficulty_submission'),
    'zip', root_dir=FINAL)
from google.colab import files
files.download(archive)
''', 'anchor-package')

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
