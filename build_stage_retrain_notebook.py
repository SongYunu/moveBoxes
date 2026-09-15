"""Fixed GitHub/T4 bootstrap, then fresh all-parcel StageACT retraining."""
import json
from pathlib import Path
from build_stage_deadline_notebook import make_notebook as base_notebook

IMPLEMENTATION_COMMIT = '711dfd5caa71acab27aa663bec48b6bc97dcd183'


def make_notebook():
    base = base_notebook(dict(project_ref=IMPLEMENTATION_COMMIT))
    original = {c['metadata'].get('id'):c for c in base['cells']}
    cells = base['cells'][:5]
    def add(kind,text,ident):
        cell = dict(cell_type=kind,metadata={'id':ident},source=text.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown','''# 새 가중치로 재학습 · 모든 다음 상자 집기 보강

01~05의 GitHub 토큰·state 데이터·T4 설치 과정은 기존과 같습니다. 기존 run을 데이터 복구에 사용하고, 모델은 별도 `_all_pick_retrain_v1` run에서 무작위 초기화합니다. 현재 실패 모델과 과거 best는 초기 가중치로 사용하지 않습니다. 같은 새 run을 다시 열면 그 run의 optimizer/scaler/RNG checkpoint만 복구합니다.

성공했던 StageACT와 같은 history16/chunk16/width128/layers2, batch64, lr1e-4, prior 직접 학습을 사용합니다. 학습 때 설정된 act_horizon=1과 실행을 일치시켜 매 step XYZ를 재추론하고 temporal ensemble4를 적용하며, gripper는 최신 예측을 바로 적용합니다. 이전 pick2/carry6 실행과 gripper 확인 필터는 새 run에 적용하지 않습니다.

배치 50%는 첫 상자 및 이후 모든 상자 집기에서 상자 순서별 균등 표집합니다. 접근·정렬·하강·집기 전체가 포함되며 같은 상자 재집기는 새 상자 순서로 세지 않습니다. 나머지는 전체 시연·운반·놓기·전환·복구 표본입니다. 난이도마다 12,000 step, 500 step마다 전체 학습 상태를 GitHub에 백업합니다.

Easy 셀의 학습 → 공식 default 평가 → 영상을 먼저 확인하세요. loss는 성능 점수가 아닙니다. 각 셀은 다음 난이도에서도 같은 학습/정책 로직을 사용하며 후보 승자 선택이나 과거 가중치 복원은 하지 않습니다. Drive를 사용하지 않습니다.
''','retrain-guide')
    helper = ''.join(original['official-smoke']['source']).split('def run_official',1)[1].rsplit('\nfor level in selected:',1)[0]
    helper = ('def run_official'+helper).replace('stage_chunk_policy:load_policy','stage_policy:load_policy')
    add('code','''# 새 run 설정 · 기존 제출본 보존
import json, os, signal, shutil, subprocess, sys
from pathlib import Path
from IPython.display import Video,display
from stage_all_pick_retrain import prepare_retrain,train_retrain,package_retrain
RETRAIN_ITERS = 12000
PICK_FOCUS_FRACTION = .5
RETRAIN_SUFFIX = '_all_pick_retrain_v1'
MAX_STEPS = 200
UPSTREAM = Path(CFG['repo_dir'])
OFFICIAL_EVAL_CONFIG = UPSTREAM/'conf/eval/default.yaml'
finished = []
SMOKE_CONFIG = Path(CFG['output_root'])/(CFG['run_name']+RETRAIN_SUFFIX+'_smoke_eval.yaml')
SMOKE_CONFIG.write_text('eval:\\n  n_episodes: 1\\n  seeds: [61000]\\n',encoding='utf-8')

def show_latest_video(level,label):
    folder = RUN_DIR/level/'integrated_official_eval'/label/'videos'
    videos = sorted(folder.rglob('*.mp4'),key=lambda path:path.stat().st_mtime)
    if not videos:
        raise FileNotFoundError(f'{folder}에 공식 평가 MP4가 없습니다.')
    display(Video(str(videos[-1]),embed=True,width=960))
print('기존 run (보존):',experiment.run_dir)
print('새 가중치 재학습:',CFG['run_name']+RETRAIN_SUFFIX)
'''+helper,'retrain-settings')
    for level in ('easy','medium','hard'):
        add('code',f'''# {level.upper()} 새 가중치 학습 · 같은 새 run에서는 전체 상태 복구
retrain = prepare_retrain(experiment,{level!r},iterations=RETRAIN_ITERS,
    focus_fraction=PICK_FOCUS_FRACTION,run_suffix=RETRAIN_SUFFIX)
RUN_DIR = Path(retrain.run_dir)
train_retrain(retrain,{level!r})
finished = [name for name in ('easy','medium','hard')
    if (retrain.run_dir/name/'training_complete.json').is_file()]
CANDIDATE = package_retrain(retrain,finished)
run_official({level!r},SMOKE_CONFIG,'smoke')
show_latest_video({level!r},'smoke')
run_official({level!r},OFFICIAL_EVAL_CONFIG,'default')
show_latest_video({level!r},'default')
print('새 run:',RUN_DIR)
''',f'retrain-{level}')
    add('code','''# 새 학습 상태 전체 + 새 제출본 PC 다운로드
from google.colab import files
archive = shutil.make_archive(str(RUN_DIR.parent/(RUN_DIR.name+'_backup')),
    'zip',root_dir=RUN_DIR.parent,base_dir=RUN_DIR.name)
files.download(archive)
submission = shutil.make_archive(str(RUN_DIR/'retrained_candidate'),
    'zip',root_dir=CANDIDATE)
files.download(submission)
print('원래 제출본과 원래 checkpoint는 보존했습니다.')
''','retrain-download')
    base['cells'] = cells
    base['metadata']['colab']['name'] = 'moveboxes_stage_all_pick_retrain_colab.ipynb'
    return base


if __name__ == '__main__':
    notebook = make_notebook()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']),cell['metadata'].get('id','cell'),'exec')
    target = Path(__file__).parent/'notebooks/moveboxes_stage_all_pick_retrain_colab.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(target)
