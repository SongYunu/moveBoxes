"""Code-only 19-cell notebook for the next-pick experiment."""
import copy
from colab_train_test_layout import TRAIN_TEST_CONFIG, make_notebook as train_test_notebook


NEXT_PICK_CONFIG = copy.deepcopy(TRAIN_TEST_CONFIG)
NEXT_PICK_CONFIG.update(
    run_name='marso_t4_next_pick_v02',
    act_horizon=4,
    chunk_candidates=[4, 8],
    focus_sampling=True,
    focus_weight=3.0,
    focus_before_steps=20,
    focus_after_steps=8,
    focus_min_grasp_steps=3,
    focus_min_release_steps=3,
)


def make_notebook(sources, config=None):
    cfg = copy.deepcopy(NEXT_PICK_CONFIG if config is None else config)
    notebook = train_test_notebook(sources, cfg)
    text = ''.join(notebook['cells'][0]['source'])
    extras = [
        '    # 다음 집기 집중 학습 · 2번째 이후 안정적인 집기 사이클 주변 시연을 3배 가중 표집',
        '    # 같은 상자의 재집기도 포함될 수 있음. sampling_audit.json에서 탐지 결과 확인.',
        '    # 비교 실험: 새 run_name을 쓰고 focus_sampling=False, act_horizon=8로 설정.',
    ]
    for key in ('focus_sampling', 'focus_weight', 'focus_before_steps', 'focus_after_steps',
                'focus_min_grasp_steps', 'focus_min_release_steps'):
        extras.append(f'    "{key}": {cfg[key]!r},')
    text = text.rsplit('}', 1)[0]+'\n'+'\n'.join(extras)+'\n}\n'
    text = text.replace('# 01 · CONFIG', '# 01 · CONFIG · Colab 2026.07 / Python 3.12 / T4\n'
        '# 이전 실험과 다른 run_name입니다. 가중 표집은 새로 학습해야 적용됩니다.\n'
        '# 모델은 [32, 64, 128], batch 32 유지. 평가 제한 200스텝 유지.')
    notebook['cells'][0]['source'] = text.splitlines(keepends=True)
    bootstrap = ''.join(notebook['cells'][1]['source'])
    bootstrap = bootstrap.replace('/content/marso_train_test_runtime', '/content/marso_next_pick_runtime')
    bootstrap = bootstrap.replace('experiment = marso_train_test.TrainTestExperiment(CFG, SOURCES)',
        'import next_pick_sampling, next_pick_diagnostics, marso_next_pick\n'
        'importlib.reload(next_pick_sampling)\nimportlib.reload(next_pick_diagnostics)\n'
        'importlib.reload(marso_next_pick)\n'
        'experiment = marso_next_pick.NextPickExperiment(CFG, SOURCES)')
    notebook['cells'][1]['source'] = bootstrap.splitlines(keepends=True)
    for cell in notebook['cells'][5:]:
        text = ''.join(cell['source'])
        text = text.replace('전체 학습\n', '전체 학습 · 다음 집기 구간 가중 표집 → Drive\n')
        text = text.replace('빠른 테스트: 8회 + 영상', '빠른 테스트: 8회 + 영상 + 다음 집기 진단')
        cell['source'] = text.splitlines(keepends=True)
    notebook['metadata']['colab']['name'] = 'marso_colab_t4_next_pick.ipynb'
    return notebook
