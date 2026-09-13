"""Small T4 baseline with explicit smoke/train/test/evaluate cells for every level."""
import copy
from colab_layout import DEFAULT_CONFIG, config_source, make_notebook as modular_notebook

TRAIN_TEST_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
TRAIN_TEST_CONFIG.update(
    run_name='marso_t4_train_test_v01',
    batch_size=32,
    pred_horizon=16,
    unet_dims=[32,64,128],
    diffusion_step_embed_dim=32,
    denoising_candidates=[16,32],
    max_checkpoints=2,
    test_episodes=8,
    test_seed_start=40000,
    test_inference_steps=16,
    test_record_video=True,
)


def make_notebook(sources, config=None):
    cfg = copy.deepcopy(TRAIN_TEST_CONFIG if config is None else config)
    notebook = modular_notebook(sources,cfg)
    config_text = config_source(cfg).replace(
        '학습 · 모델 크기는 기존 T4 개선본과 동일','학습 · T4용 소형 모델, batch 32')
    extras = ['    # 빠른 테스트 · 최종 평가와 별도 저장']
    for key in ('test_episodes','test_seed_start','test_inference_steps','test_record_video'):
        extras.append(f'    "{key}": {cfg[key]!r},')
    config_text = config_text.rsplit('}',1)[0]+'\n'+'\n'.join(extras)+'\n}\n'
    notebook['cells'][0]['source'] = config_text.splitlines(keepends=True)
    bootstrap = ''.join(notebook['cells'][1]['source'])
    bootstrap = bootstrap.replace("MODULE_DIR = Path('/content/marso_runtime')", "MODULE_DIR = Path('/content/marso_train_test_runtime')")
    bootstrap = bootstrap.replace('experiment = marso_experiment.Experiment(CFG, SOURCES)',
        'import marso_train_test\nimportlib.reload(marso_train_test)\nexperiment = marso_train_test.TrainTestExperiment(CFG, SOURCES)')
    notebook['cells'][1]['source'] = bootstrap.splitlines(keepends=True)
    notebook['cells'] = notebook['cells'][:5]

    def cell(source):
        i=len(notebook['cells'])+1
        notebook['cells'].append(dict(cell_type='code',id=f'cell-{i:02d}',metadata={},
                                      execution_count=None,outputs=[],source=source.splitlines(keepends=True)))

    for level in ('easy','medium','hard'):
        i=len(notebook['cells'])+1
        cell(f'# {i:02d} · {level.upper()} 동작 확인: 별도 폴더에서 100 iteration + 2회 시험\n'
             '# 성능 점수가 아니라 학습·모델 로딩·환경 실행 확인입니다.\n'
             f'experiment.smoke("{level}")\n')
        cell(f'# {i+1:02d} · {level.upper()} 전체 학습\nexperiment.train("{level}")\n')
        cell(f'# {i+2:02d} · {level.upper()} 빠른 테스트: 8회 + 영상 → test_metrics.json\n'
             f'experiment.test("{level}")\n')
        cell(f'# {i+3:02d} · {level.upper()} 설정 비교 + 최종 100회 평가 → metrics.json\n'
             '# 시간이 오래 걸릴 수 있습니다. 필요할 때 이 셀만 따로 실행할 수 있습니다.\n'
             f'experiment.evaluate("{level}")\n')
    cell('# 18 · Drive의 테스트·최종 평가 결과와 영상 보기\n'
         'experiment.show_tests(videos=True)\nexperiment.show_results(videos=True)\n')
    cell('# 19 · 최종 평가까지 끝난 모델을 Drive에 패키징\nexperiment.package()\n')
    notebook['metadata']['colab']['name']='marso_colab_t4_train_test.ipynb'
    return notebook
