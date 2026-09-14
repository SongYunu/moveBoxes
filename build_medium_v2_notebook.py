"""Code-only Colab notebook for a separate, resource-bounded Medium v2 run."""
import json
from pathlib import Path
from build_medium_notebook import CONFIG as V1_CONFIG, make_notebook as v1_notebook

CONFIG=dict(V1_CONFIG,run_name='moveboxes_medium_deadline_v2',source_run_name='moveboxes_medium_lab_v1',
    block_iters=500,max_blocks=6,batch_size=32,lr=3e-5,warmup_steps=50,validation_batches=4,
    first_pick_fraction=0.,position_noise=0.,ensemble_candidates=[4],ensemble_window=4,
    plateau_blocks=2,development_episodes=8,test_episodes=8,final_episodes=100,
    tuning_seed_start=52000,test_seed_start=42000,eval_seed_start=72000,
    deadline_episodes=16,deadline_max_attempts=40,teacher_gains=[1.25,1.75],pilot_episodes=2,
    pilot_seed_start=120000,deadline_seed_start=121000,deadline_noise_std=.08,
    deadline_noise_probability=.1,collection_seconds_per_call=600)


def make_notebook(config=None):
    cfg=dict(CONFIG,**(config or {}));nb=v1_notebook(cfg)
    groups=[('저장 · v1 최고 모델에서 시작, v2 결과는 별도 Release',
        ['run_name','source_run_name','warm_start','profile','github_repository','project_dir','project_ref','output_root']),
        ('T4 / Python 3.12 / Colab 2026.07',
        ['repo_dir','repo_url','repo_commit','data_dir','data_source','download_cache','packages']),
        ('시연 · 199행동 안에 네 상자가 성공한 실제 시연만 사용',
        ['teacher_gains','pilot_episodes','deadline_episodes','deadline_max_attempts','collection_seconds_per_call',
         'pilot_seed_start','deadline_seed_start','deadline_noise_std','deadline_noise_probability']),
        ('보정 · 500회마다 평가, 최대 3,000회, 2구간 개선 없으면 종료',
        ['block_iters','max_blocks','batch_size','lr','amp','plateau_blocks','target_accuracy','success_streak']),
        ('기존 모델 구조 유지 / 집기 접촉 구간은 모든 상자에서 표집',
        ['seed','num_demos','history','chunk_size','width','heads','layers','latent_dim']),
        ('개발 / 테스트 / 최종 평가 시드 분리',
        ['development_episodes','test_episodes','final_episodes','tuning_seed_start','test_seed_start','eval_seed_start','max_episode_steps']),
        ('실행 / 출력',
        ['ensemble_candidates','ensemble_window','temporal_decay','gate_threshold','stage_threshold','console_interval_seconds','team'])]
    lines=['# 01 · MEDIUM v2 CONFIG','CFG = {']
    for title,keys in groups:
        lines+=['    # '+title]+[f'    {key!r}: {cfg[key]!r},' for key in keys]+['']
    lines+=['}']
    nb['cells'][0]['source']=('\n'.join(lines)+'\n').splitlines(keepends=True)
    bootstrap=''.join(nb['cells'][1]['source'])
    bootstrap=bootstrap.replace('from medium_lab import MediumLab, source_bundle\nexperiment = MediumLab(CFG, source_bundle())',
        "for name in ('build_medium_v2_notebook','medium_v2'):\n"
        "    if name in sys.modules:\n        importlib.reload(sys.modules[name])\n"
        "from build_medium_v2_notebook import CONFIG as V2_DEFAULTS\n"
        "PROJECT_COMMIT = CFG['project_commit']\n"
        "CFG = dict(V2_DEFAULTS, **USER_CONFIG)\n"
        "CFG['project_commit'] = PROJECT_COMMIT\n"
        "from medium_v2 import MediumV2, source_bundle\nexperiment = MediumV2(CFG, source_bundle())")
    # Preserve the user's overrides before v1 fills its own hidden defaults.
    bootstrap='USER_CONFIG = dict(CFG)\n'+bootstrap
    nb['cells'][1]['source']=bootstrap.splitlines(keepends=True)
    nb['cells'][2]['source']=['# 03 · GitHub 인증/복원\n','experiment.connect()\n','_ = experiment.report()\n']
    nb['cells']=nb['cells'][:4]
    tasks=[('05 · GPU 확인 + 기존 최고 모델 가져오기','experiment.check_runtime()\nexperiment.prepare()'),
        ('06 · 빠른 시연 소규모 시험 · 실패하면 학습 진입 차단','_ = experiment.collect_deadline("pilot")'),
        ('07 · 제한 내 성공 시연 수집 · 시간 예산으로 멈추면 같은 셀 재실행','_ = experiment.collect_deadline("collect")'),
        ('08 · 보정 학습 + 같은 시드 평가 · 중단하면 같은 셀로 재개','_ = experiment.run_blocks()'),
        ('09 · 최고 모델 별도 테스트 + 영상 + 행동 기록','experiment.test("medium")\nexperiment.diagnose("medium")'),
        ('10 · 구간별 성적 확인','_ = experiment.report()'),
        ('11 · 선택한 최고 모델 최종 100회 평가 · 부분 점수도 측정','_ = experiment.final_evaluation()'),
        ('12 · 최종 평가 모델 패키징','_ = experiment.package()')]
    for i,(title,code) in enumerate(tasks,5):
        nb['cells'].append(dict(cell_type='code',execution_count=None,outputs=[],metadata={'id':f'medium-v2-{i}'},
            source=(f'# {title}\n{code}\n').splitlines(keepends=True)))
    nb['metadata']['colab']['name']='moveboxes_medium_v2_colab.ipynb'
    return nb


if __name__=='__main__':
    nb=make_notebook()
    for i,c in enumerate(nb['cells']):compile(''.join(c['source']),str(i),'exec')
    path=Path(__file__).parent/'notebooks/moveboxes_medium_v2_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path.name,len(nb['cells']),'code cells')
