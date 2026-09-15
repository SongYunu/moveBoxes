"""Build a no-Drive Colab for PICK-only group-relative residual RL."""
import json
from pathlib import Path

from build_stage_anchor_notebook import make_notebook as anchor_notebook


IMPLEMENTATION_COMMIT = 'b67d8ba6ddcf1d514a33ca2e82859a534720dc47'


def make_notebook():
    nb = anchor_notebook()
    cells = nb['cells'][:8]
    config = ''.join(cells[0]['source'])
    old_ref = next(line.split("'")[3] for line in config.splitlines() if "'project_ref':" in line)
    config = config.replace(old_ref, IMPLEMENTATION_COMMIT)
    config = config.replace("'run_name': 'moveboxes_stage_anchor_v1'",
                            "'run_name': 'moveboxes_stage_pick_residual_v1'")
    cells[0]['source'] = config.splitlines(keepends=True)
    cells[5]['source'] = '''# Stage ACT 앵커 + PICK Residual Group RL

검증된 난이도별 Stage ACT 앵커를 그대로 복원하고 **전체 앵커를 동결**합니다. 새 정책은 `PICK` 단계의 XYZ에만 작은 상태 기반 residual을 더합니다. residual의 마지막 층은 0으로 초기화되므로 학습 전 동작은 앵커와 정확히 같습니다.

각 초기 배치 seed마다 4개의 stochastic 집기 후보를 실행하고, 같은 배치 안에서 실제 `success_count`가 더 높은 후보에만 상대 advantage를 줍니다. critic과 demonstration target은 사용하지 않습니다. 한 집기 시도의 마지막 12 step만 업데이트하므로 운반·놓기와 긴 대기 동작이 집기 보정 gradient를 오염시키지 않습니다.

8회마다 공식 `eval.py`를 실행합니다. 최고 공식 점수는 별도 파일로 보존하고, 3번 연속 개선이 없으면 자동으로 멈춥니다. Drive는 사용하지 않으며 iteration checkpoint와 optimizer는 GitHub Release에 복구됩니다.
'''.splitlines(keepends=True)
    setup = ''.join(cells[6]['source'])
    setup = setup.replace('prepare_success_rl,\n                                   train_success_rl',
        'prepare_pick_residual_rl,\n                                   train_pick_residual_rl')
    setup = setup.replace('MEDIUM_RL_ITERATIONS = 512 # 양의 정수: 상한 없음. 늘린 뒤 재실행하면 이어서 학습\n'
                          'HARD_RL_ITERATIONS = 64    # 양의 정수: 상한 없음\n'
                          'RL_EVAL_EVERY = 32',
                          'MEDIUM_GROUP_ITERATIONS = 64\nRL_EVAL_EVERY = 8\nEARLY_STOP_EVALS = 3')
    setup = setup.replace("run_suffix='_anchor_success_base_v1'", "run_suffix='_anchor_pick_residual_base_v1'")
    setup = setup.replace("'success_rl_official_eval'", "'pick_residual_official_eval'")
    setup = setup.replace("OFFICIAL = UPSTREAM/'conf/eval/default.yaml'",
        "OFFICIAL_DEFAULT = UPSTREAM/'conf/eval/default.yaml'\n"
        "SELECTION = Path(CFG['output_root'])/'pick_residual_selection_eval.yaml'\n"
        "SELECTION.write_text('eval:\\n  n_episodes: 8\\n  seeds: '+"
        "json.dumps(list(range(63000, 63008)))+'\\n', encoding='utf-8')")
    setup = setup.replace('def run_official(candidate, level, label):',
                          'def run_official(candidate, level, label, eval_config=SELECTION):')
    setup = setup.replace("'eval_config='+str(OFFICIAL)", "'eval_config='+str(eval_config)")
    cells[6]['source'] = setup.splitlines(keepends=True)
    cells[7]['source'] = '''# 07 · Medium 학습 전 8-seed 선택 평가 + 영상
BASELINE_RESULTS = {'medium':run_official(BASELINE, 'medium', 'anchor_selection')}
print(json.dumps(BASELINE_RESULTS, indent=2))
'''.splitlines(keepends=True)

    def add(kind, source, ident):
        cell = dict(cell_type=kind, metadata={'id':ident}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''## Medium PICK residual 학습

`GROUP RL` 로그의 `sorted`는 해당 iteration의 16개 rollout이 분류한 상자 수이고, 모델 선택 점수는 매 8회 뒤 출력되는 공식 `SORT ACCURACY`입니다. 모든 동일-seed 후보 점수가 같으면 update를 건너뜁니다. 중단되어도 이 셀을 다시 실행하면 마지막 완전한 iteration에서 재개합니다.
''', 'pick-residual-guide')

    add('code', '''# 09 · Medium PICK residual · 8회마다 공식 평가 · 3회 미개선 자동 중단
def evaluate_group_round(rl_exp, level, stop, best):
    boundary = rl_exp.run_dir/level/'checkpoints'/f'iteration_{stop:04d}.pt'
    if not boundary.is_file():
        raise FileNotFoundError(f'라운드 체크포인트가 없습니다: {boundary}')
    candidate = package(rl_exp, checkpoint_overrides={level:boundary},
                        folder_name=f'{level}_pick_residual_round_{stop:02d}')
    result = run_official(candidate, level, f'pick_residual_round_{stop:02d}')
    result.update(iteration=stop, source='pick_residual_group_rl')
    history_path = rl_exp.run_dir/level/'official_rounds.json'
    history = json.loads(history_path.read_text()) if history_path.is_file() else []
    history = [item for item in history if item.get('iteration') != stop]
    history.append({k:v for k,v in result.items() if k != 'checkpoint'})
    history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    improved = result['score'] > best['score']
    if improved:
        promoted = rl_exp.run_dir/level/'checkpoints'/'official_best.pt'
        shutil.copy2(result['checkpoint'], promoted)
        result['checkpoint'] = str(promoted)
        best = result
        print(f'{level} {stop}회: 새 최고 공식 점수 {result["score"]:.3%}')
    else:
        print(f'{level} {stop}회: {result["score"]:.3%}; 현재 최고 {best["score"]:.3%} 유지')
    try:
        rl_exp.sync_level(level)
    except Exception as error:
        print('GitHub 백업 지연; 로컬 최고 체크포인트는 유지합니다:', error)
    return best, improved

def evaluation_stops(total, every=RL_EVAL_EVERY):
    stops = list(range(every, total+1, every))
    if not stops or stops[-1] != total:
        stops.append(total)
    return stops

medium_group = prepare_pick_residual_rl(
    anchor_exp, 'medium', iterations=MEDIUM_GROUP_ITERATIONS,
    num_envs=16, group_size=4, lr=2e-5, xyz_std=.04, pick_credit_steps=12)
MEDIUM_BEST = dict(BASELINE_RESULTS['medium'], iteration=0, source='anchor')
stale_evals = 0
for stop in evaluation_stops(MEDIUM_GROUP_ITERATIONS):
    train_pick_residual_rl(medium_group, 'medium', until_iteration=stop)
    MEDIUM_BEST, improved = evaluate_group_round(medium_group, 'medium', stop, MEDIUM_BEST)
    stale_evals = 0 if improved else stale_evals+1
    if stale_evals >= EARLY_STOP_EVALS:
        print(f'공식 점수 {EARLY_STOP_EVALS}회 연속 미개선: 최고 체크포인트를 유지하고 중단합니다.')
        break
MEDIUM_USE_GROUP = MEDIUM_BEST['source'] == 'pick_residual_group_rl'
print('Medium 최종 선택:', MEDIUM_BEST)
''', 'pick-residual-medium')

    add('code', '''# 10 · 검증 앵커를 기본 제출로 고정 · 전체 난이도 재검증 · ZIP
# Medium 8-seed 34.4%를 기록한 모델은 별도 RL 모델이 아니라 이 anchor.pt입니다.
selected = {}
FINAL = package(anchor_exp, folder_name='final_verified_anchor_candidate')
manifest = json.loads((FINAL/'manifest.json').read_text(encoding='utf-8'))
assert manifest['levels']['medium']['selection'] == 'anchor'
assert manifest['levels']['medium']['checkpoint_sha256'] == ANCHORS['medium']['sha256']
FINAL_RESULTS = {level:run_official(FINAL, level, 'final_default', OFFICIAL_DEFAULT)
                 for level in ('easy','medium','hard')}
print('DEFAULT Medium = verified block_02.pt anchor · selection score 34.4% on seeds 63000..63007')
print(json.dumps({'selected':selected,
                  'official_results':FINAL_RESULTS}, indent=2))
archive = shutil.make_archive(str(RUN_DIR/'stage_act_verified_anchor_submission'),
                              'zip', root_dir=FINAL)
demo_dir = Path(BASELINE_RESULTS['medium']['log']).parent/'videos'
demo_videos = sorted(demo_dir.rglob('*.mp4'), key=lambda p:p.stat().st_mtime)
if demo_videos:
    print('Medium ANCHOR DEMO · 8-seed selection SORT ACCURACY 34.4%')
    display(Video(str(demo_videos[-1]), embed=True, width=900))
from google.colab import files
files.download(archive)
''', 'pick-residual-package')

    nb['cells'] = cells
    nb['metadata']['colab']['name'] = 'moveboxes_stage_pick_residual_colab.ipynb'
    return nb


if __name__ == '__main__':
    notebook = make_notebook()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['metadata'].get('id','cell'), 'exec')
    target = Path(__file__).parent/'notebooks/moveboxes_stage_pick_residual_colab.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(target)
