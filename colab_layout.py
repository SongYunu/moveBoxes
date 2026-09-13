"""Code-only notebook layout; used locally and to save an independent Drive copy."""
import base64
import json
import zlib

DEFAULT_CONFIG = {
    'run_name': 'marso_t4_chunk_v01',
    'profile': 'benchmark',
    'output_root': '/content/drive/MyDrive/marso_benchmark',
    'data_source': '/content/drive/MyDrive/marso-hack-berlin-2026-robot-parcel-sorting-challenge.zip',
    'repo_dir': '/content/berlin-marso-hackathon',
    'data_dir': '/content/marso_data',
    'repo_url': 'https://github.com/marso-robotics/berlin-marso-hackathon.git',
    'repo_commit': '6048f33217f26ae39009a812f53c81171517f393',
    'packages': ['mani-skill==3.0.1', 'sapien==3.0.3', 'diffusers==0.38.0',
                 'hydra-core', 'omegaconf', 'gymnasium', 'tyro', 'h5py', 'kagglehub',
                 'tensorboard', 'matplotlib', 'transforms3d', 'imageio[ffmpeg]'],
    'seed': 42,
    'num_demos': None,
    'batch_size': 128,
    'lr': 1e-4,
    'total_iters': {'easy': 30000, 'medium': 50000, 'hard': 60000},
    'obs_horizon': 2,
    'pred_horizon': 32,
    'act_horizon': 8,
    'unet_dims': [64, 128, 256],
    'diffusion_step_embed_dim': 64,
    'n_groups': 8,
    'eval_freq': 10000,
    'save_freq': 10000,
    'log_freq': 500,
    'train_eval_episodes': 4,
    'num_eval_envs': 1,
    'max_episode_steps': {'easy': 200, 'medium': 200, 'hard': 200},
    'tuning_episodes': 8,
    'tuning_seed_start': 20000,
    'chunk_candidates': [4, 8],
    'denoising_candidates': [16, 32, 100],
    'max_checkpoints': 3,
    'benchmark_episodes': 100,
    'eval_seed_start': 30000,
    'record_eval_video': True,
    'console_interval_seconds': 30,
    'team': 'my-team',
}


def config_source(cfg):
    groups = [
        ('저장 경로 · 기존 결과를 불러오려면 같은 run_name',
         ['run_name', 'profile', 'output_root', 'data_source']),
        ('공식 코드 · 런타임 2026.07 / T4',
         ['repo_dir', 'data_dir', 'repo_url', 'repo_commit', 'packages']),
        ('학습 · 모델 크기는 기존 T4 개선본과 동일',
         ['seed', 'num_demos', 'batch_size', 'lr', 'total_iters']),
        ('모델 · 새 학습에 적용; 기존 checkpoint 구조는 저장 설정 사용',
         ['obs_horizon', 'pred_horizon', 'act_horizon', 'unet_dims',
          'diffusion_step_embed_dim', 'n_groups']),
        ('학습 중 평가 / 저장',
         ['eval_freq', 'save_freq', 'log_freq', 'train_eval_episodes', 'num_eval_envs']),
        ('평가 · 난이도별 제한 200스텝 유지',
         ['max_episode_steps', 'tuning_episodes', 'tuning_seed_start', 'chunk_candidates',
          'denoising_candidates', 'max_checkpoints', 'benchmark_episodes', 'eval_seed_start']),
        ('출력', ['record_eval_video', 'console_interval_seconds', 'team']),
    ]
    lines = ['# 01 · CONFIG', 'CFG = {']
    for title, keys in groups:
        lines.append(f'    # {title}')
        for key in keys:
            if key == 'packages':
                lines.append('    "packages": [')
                lines.extend('        '+repr(v)+',' for v in cfg[key])
                lines.append('    ],')
            else:
                lines.append(f'    "{key}": {cfg[key]!r},')
        lines.append('')
    lines.append('}')
    return '\n'.join(lines)+'\n'


def make_notebook(sources, config=None):
    cfg = dict(DEFAULT_CONFIG if config is None else config)
    payload = base64.b64encode(zlib.compress(json.dumps(sources).encode('utf-8'))).decode('ascii')
    bootstrap = '''# 02 · 공통 모듈 로드 (이 셀은 접혀 있습니다)
import base64, json, sys, zlib
from pathlib import Path

SOURCES = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
MODULE_DIR = Path('/content/marso_runtime')
MODULE_DIR.mkdir(parents=True, exist_ok=True)
for name, source in SOURCES.items():
    (MODULE_DIR / name).write_text(source, encoding='utf-8')
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
import importlib
import marso_experiment
importlib.reload(marso_experiment)
experiment = marso_experiment.Experiment(CFG, SOURCES)
print('공통 학습·평가 모듈 준비 완료')
'''.replace('SOURCES =', 'PAYLOAD = '+repr(payload)+'\nSOURCES =', 1)
    sources_for_cells = [
        config_source(cfg), bootstrap,
        '# 03 · Drive 연결 / 저장된 결과 복원 / 실행 노트북을 Drive에 자동 저장\n'
        'from google.colab import drive\ndrive.mount("/content/drive")\n'
        'experiment.connect()\nexperiment.show_results()\n',
        '# 04 · 새 런타임마다 환경 설치\nexperiment.install()\n',
        '# 05 · 데이터 준비 / GPU와 정책 실행 확인\nexperiment.prepare_data()\nexperiment.check_runtime()\n',
    ]
    for index, level in enumerate(('easy', 'medium', 'hard')):
        n = 6 + index*2
        sources_for_cells += [
            f'# {n:02d} · {level.upper()} 학습 → Drive/{level}/checkpoints\nexperiment.train("{level}")\n',
            f'# {n+1:02d} · {level.upper()} 평가 → Drive/{level}/metrics.json\nexperiment.evaluate("{level}")\n',
        ]
    sources_for_cells += [
        '# 12 · 난이도별 점수와 영상 불러오기 (저장된 파일만 읽음)\n'
        'experiment.show_results(videos=True)\n',
        '# 13 · 준비된 모델을 Drive에 패키징\nexperiment.package()\n',
    ]
    cells = [dict(cell_type='code', id=f'cell-{i+1:02d}', metadata={},
                  execution_count=None, outputs=[], source=src.splitlines(keepends=True))
             for i, src in enumerate(sources_for_cells)]
    cells[1]['metadata'] = {'cellView': 'form', 'source_hidden': True}
    return dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
        'accelerator': 'GPU', 'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
        'language_info': {'name':'python'},
        'colab': {'name':'marso_colab_t4_modular.ipynb', 'provenance':[]},
    })
