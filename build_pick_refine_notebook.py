"""Same GitHub/data bootstrap, followed by protected pick-focused refinement."""
import json
from pathlib import Path
from build_stage_deadline_notebook import make_notebook as base_notebook


def make_notebook():
    base = base_notebook()
    original = {c['metadata'].get('id'):c for c in base['cells']}
    cells = base['cells'][:5]

    def add(kind, source, ident):
        cell = dict(cell_type=kind, metadata={'id':ident}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)

    add('markdown', '''# 현재 제출본을 보존하는 집기 조준 보정

01~05의 GitHub 토큰·데이터·T4 준비 과정은 기존과 같습니다. 같은 현재 run을 복구한 후 candidate를 재생성합니다. 부모 run의 학습 설정은 바꾸지 않습니다.

기본은 추가 학습 비활성화 상태입니다. 실행 수정본을 먼저 평가합니다. 학습을 활성화하면 Easy 6,000 step, 기존 학습률 1e-4로 보정합니다. `REFINE_LEVELS`에 Medium/Hard를 추가하면 같은 정책 로직을 사용합니다. 배치 50%는 첫 상자와 이후 모든 상자의 집기 구간에서 상자 순서별로 균등 표집하고, 나머지는 기존 전체 시연·전환·복구 데이터에서 학습합니다. 같은 상자 재집기는 새 상자로 세지 않습니다. Carry/place도 함께 학습하지만 각 단계의 개선은 실제 영상과 공식 점수로 확인해야 합니다.

실행 horizon과 200-step episode 제한은 유지합니다. 최종 제출본으로 바꾸기 전 공식 평가와 영상을 확인하세요. 원래 후보에는 덮어쓰지 않습니다.
''','refine-guide')
    cells.append(original['integrated-candidate'])
    cells.append(original['run-backup-download'])
    add('code', '''# 집기 보정 설정
import os, subprocess, sys
from stage_pick_finetune import prepare_pick_finetune, train_pick_finetune, package_pick_finetune
REFINE_LEVELS = ['easy']  # 필요하면 ['easy','medium','hard']
REFINE_ITERS = 6000
REFINE_LR_FACTOR = 1.
PICK_FOCUS_FRACTION = .5
RUN_FINE_TUNE = False  # 실행 수정본의 공식 평가와 영상을 확인한 뒤 직접 활성화
UPSTREAM = Path(CFG['repo_dir'])
refined = {}
''','refine-settings')
    add('code', '''# 기존 packaged policy의 집기 XYZ/stage/grasp 기록 (점수 아님)
for level in REFINE_LEVELS:
    child_env = dict(os.environ)
    child_env['PYTHONPATH'] = str(CANDIDATE)+os.pathsep+str(UPSTREAM)+os.pathsep+child_env.get('PYTHONPATH','')
    subprocess.run([sys.executable,str(PROJECT/'ver2/stages/stage_pick_diagnose.py'),
        '--checkpoint',str(CANDIDATE/'checkpoints'/level/'model.pt'),
        '--config-dir',str(UPSTREAM/'conf'),'--level',level,
        '--output',str(RUN_DIR/level/'pick_alignment_trace.json')],
        cwd=UPSTREAM,env=child_env,check=True)
''','refine-diagnose')
    add('code', '''# 별도 run에서 보정 · 재실행하면 동일 job의 optimizer/RNG 복구
for level in REFINE_LEVELS if RUN_FINE_TUNE else []:
    child = prepare_pick_finetune(experiment,CANDIDATE,level,iterations=REFINE_ITERS,
        lr_factor=REFINE_LR_FACTOR,first_pick_fraction=PICK_FOCUS_FRACTION,
        pick_sampling='all_picks',run_suffix='_all_pick_refine_v1')
    refined[level] = child
    train_pick_finetune(child,level)
if not RUN_FINE_TUNE:
    print('추가 학습은 비활성화 상태입니다. 현재 checkpoint로 실행 수정본부터 평가합니다.')
''','refine-train')
    add('code', '''# 한 integrated candidate에 보정된 난이도의 가중치만 반영
REFINED_CANDIDATE = CANDIDATE
for level,child in refined.items():
    REFINED_CANDIDATE = package_pick_finetune(child,CANDIDATE,level)
print('원래 제출본:',CANDIDATE)
print('별도 보정 제출본:',REFINED_CANDIDATE)
''','refine-package')
    # Reuse the official evaluator lifecycle, not a custom metric or selection.
    helper = ''.join(original['official-smoke']['source']).rsplit('\nfor level in selected:',1)[0]
    helper = helper.replace('RUN_DIR', 'REFINED_RUN_DIR').replace('CANDIDATE','REFINED_CANDIDATE')
    add('code', '''# 같은 공식 default.yaml 평가 · 같은 horizon, metric, episode 제한
REFINED_RUN_DIR = next(iter(refined.values())).run_dir if refined else RUN_DIR
'''+helper+'''
from IPython.display import Video,display
for level in REFINE_LEVELS:
    run_official(level,UPSTREAM/'conf/eval/default.yaml','default')
    folder = REFINED_RUN_DIR/level/'integrated_official_eval/default/videos'
    videos = sorted(folder.rglob('*.mp4'),key=lambda path:path.stat().st_mtime)
    if videos:
        display(Video(str(videos[-1]),embed=True,width=960))
''','refine-official-eval')
    add('code', '''# 보정 run 전체 PC 백업 (원래 run의 ZIP과 별도)
from google.colab import files
archive = shutil.make_archive(str(REFINED_RUN_DIR.parent/(REFINED_RUN_DIR.name+'_backup')),
    'zip',root_dir=REFINED_RUN_DIR.parent,base_dir=REFINED_RUN_DIR.name)
print(archive)
files.download(archive)
''','refine-backup')
    base['cells'] = cells
    base['metadata']['colab']['name'] = 'moveboxes_stage_pick_refine_colab.ipynb'
    return base


if __name__ == '__main__':
    notebook = make_notebook()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']),cell['metadata'].get('id','cell'),'exec')
    target = Path(__file__).parent/'notebooks/moveboxes_stage_pick_refine_colab.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(target)
