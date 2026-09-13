"""Add a ver2 notebook; never regenerates or replaces the existing DP notebook."""
import json
from pathlib import Path
from build_github_notebook import CONFIG as OLD_CONFIG, make_notebook as old_notebook

CONFIG = dict(OLD_CONFIG, run_name='moveboxes_act_ver2', history=16, chunk_size=16,
    width=128, heads=4, layers=2, latent_dim=16, batch_size=64, lr=1e-4,
    total_iters=dict(easy=12000,medium=20000,hard=30000), save_freq=2000, warmup_steps=500,
    kl_weight=.001, position_noise=.001, validation_batches=8, amp=True,
    temporal_decay=.25, ensemble_window=4, ensemble_candidates=[1,4], test_inference_steps=1)

GROUPS = [
    ('저장 · ver2는 기존 DP 결과와 별도 Release에 저장', ['run_name','profile','github_repository','project_dir','project_ref','output_root']),
    ('데이터 / Colab 2026.07 · Python 3.12 · T4', ['repo_dir','repo_url','repo_commit','data_dir','data_source','download_cache','packages']),
    ('학습 · 시뮬레이터 없이 CPU 데이터 + GPU 모델', ['seed','num_demos','batch_size','lr','total_iters','amp']),
    ('작은 State ACT · 기존 DP 체크포인트 사용 불가', ['history','chunk_size','width','heads','layers','latent_dim']),
    ('검증 / 중단 복구 / 작은 관측 위치 증강', ['save_freq','warmup_steps','validation_batches','kl_weight','position_noise']),
    ('실행 · 매 스텝 재계획, 최근 XYZ 예측 평균, 집게는 최신 예측', ['temporal_decay','ensemble_window','ensemble_candidates']),
    ('빠른 테스트 / 최종 평가 · 시드 분리, 기존 200스텝 유지', ['test_episodes','test_seed_start','test_record_video','tuning_episodes','tuning_seed_start','benchmark_episodes','eval_seed_start','max_episode_steps','record_eval_video']),
    ('출력', ['console_interval_seconds','team']),
]


def make_notebook(config=None):
    cfg = dict(CONFIG, **(config or {}))
    nb = old_notebook(cfg)
    lines = ['# 01 · VER 2 CONFIG · 기존 코드/모델과 별도 실험', 'CFG = {']
    for title, keys in GROUPS:
        lines += ['    # '+title]
        for key in keys:
            lines.append(f'    {key!r}: {cfg[key]!r},')
        lines.append('')
    lines.append('}')
    nb['cells'][0]['source'] = ('\n'.join(lines)+'\n').splitlines(keepends=True)
    bootstrap = ''.join(nb['cells'][1]['source'])
    bootstrap = bootstrap.replace('from marso_github import GitHubExperiment, source_bundle\nexperiment = GitHubExperiment(CFG, source_bundle())',
        "sys.path.insert(0, str(PROJECT/'ver2'))\n"
        "for name in ('act_v2_model','act_v2_data','act_v2_policy','act_v2_eval','act_v2_experiment','build_act_v2_notebook'):\n"
        "    if name in sys.modules:\n        importlib.reload(sys.modules[name])\n"
        'from act_v2_experiment import ActV2Experiment, source_bundle\nexperiment = ActV2Experiment(CFG, source_bundle())')
    nb['cells'][1]['source'] = bootstrap.splitlines(keepends=True)
    for c in nb['cells'][5:]:
        text = ''.join(c['source']).replace('다음 집기 구간 가중 표집', 'State ACT ver2 · 저장 상태에서 이어 학습')
        text = text.replace('설정 비교 + 최종', '이동 예측 평균화 비교 + 최종')
        c['source'] = text.splitlines(keepends=True)
    nb['metadata']['colab']['name'] = 'moveboxes_ver2_colab.ipynb'
    return nb


if __name__ == '__main__':
    nb = make_notebook()
    for i,c in enumerate(nb['cells']):
        compile(''.join(c['source']), f'ver2-cell-{i+1}', 'exec')
    path = Path(__file__).parent/'notebooks/moveboxes_ver2_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path.name, len(nb['cells']), 'code cells')
