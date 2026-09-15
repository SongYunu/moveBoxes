"""Restore the current run and evaluate the repaired execution without training."""
import json
from pathlib import Path
from build_stage_deadline_notebook import make_notebook as base_notebook


def make_notebook():
    base = base_notebook()
    original = {c['metadata'].get('id'):c for c in base['cells']}
    guide = dict(cell_type='markdown',metadata={'id':'execution-check-guide'},source=['''# 현재 checkpoint의 실행 수정 확인

01~05는 기존 GitHub 토큰·데이터·T4 준비 과정과 동일합니다. 기존 run의 latest checkpoint를 복구합니다. 이 노트북은 학습과 시연 수집을 실행하지 않습니다.

기존 `integrated_candidate`는 보존하고 `integrated_candidate_v2`에 같은 가중치를 복사합니다. XYZ 실행 배분 pick=2/carry=6/place=2/done=1을 유지하고, 그리퍼는 매 step 현재 관측으로 다시 예측합니다. 집기·놓기 명령을 늦춘 2-step 확인 필터는 끕니다.

공식 smoke → default 평가와 각 난이도 영상을 확인하세요. CPU 시연 재생에서는 명령 지연이 제거됐지만 실제 분류 성능은 아직 확인되지 않았습니다. 평가 결과를 확인한 뒤 추가 학습을 결정합니다. Drive와 승자 선택은 사용하지 않습니다.
'''])
    ids = ('integrated-candidate','integrated-sanity','video-player','official-smoke','video-smoke',
           'official-default','video-default','run-backup-download','candidate-download')
    base['cells'] = base['cells'][:5]+[guide]+[original[i] for i in ids]
    reference = dict(cell_type='code',execution_count=None,outputs=[],metadata={'id':'reference-easy-check'},
        source=['''# 성공 기록이 있는 Easy Stage ACT · 비교 검사 전용, 현재 candidate에는 반영하지 않음
from stage_reference_check import prepare_reference_check
REFERENCE_ONLY = prepare_reference_check(CFG['github_repository'],CANDIDATE,RUN_DIR/'reference_only')
run_official('easy',OFFICIAL_EVAL_CONFIG,'reference_easy',candidate=REFERENCE_ONLY)
show_official_video('reference_easy','easy')
print('현재 제출본:',CANDIDATE)
print('참고 모델은 제출본에 반영하지 않았습니다.')
'''])
    base['cells'].insert(-2,reference)
    base['metadata']['colab']['name'] = 'moveboxes_stage_execution_check_colab.ipynb'
    return base


if __name__ == '__main__':
    notebook = make_notebook()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']),cell['metadata'].get('id','cell'),'exec')
    target = Path(__file__).parent/'notebooks/moveboxes_stage_execution_check_colab.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(target)
