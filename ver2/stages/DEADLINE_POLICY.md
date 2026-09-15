# State Stage ACT 단일 deadline 학습·평가

[Colab에서 열기](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/stage-act-chunk-compare/notebooks/moveboxes_stage_deadline_colab.ipynb)

## 전체 흐름

```text
GitHub 토큰 인증
  → 같은 run_name의 현재 학습 상태 복원
  → GitHub Release의 state dataset 다운로드·SHA256 검증
  → recovery 시연 수집
  → State Stage ACT 학습
  → 현재 run checkpoint 복사
  → stage-aware action buffer + transition replan + learned gripper hysteresis
  → 공식 eval.py smoke/default 평가
  → candidate ZIP
```

Google Drive는 사용하지 않습니다. dataset cache, simulator, 학습 결과의 로컬 경로는 모두 `/content` 아래에 있습니다. 런타임이 사라지면 cache는 다시 다운로드하지만, 학습 결과는 GitHub Release에서 복원합니다.

## GitHub 인증과 중단 복구

03 셀은 다음 순서로 토큰을 읽습니다.

1. `GH_TOKEN` 환경변수
2. Colab Secrets의 `GH_TOKEN`
3. 화면에 표시되는 비공개 입력창

Fine-grained token은 `SongYunu/moveBoxes` 저장소의 **Contents: Read and write** 권한이 필요합니다. 토큰은 런타임 환경변수에만 둡니다.

기본 `run_name`은 `moveboxes_stage_chunk_deadline_v1`입니다. 이 이름의 Release에는 이 실행에서 만든 상태만 저장됩니다. 과거 Easy/Medium/Hard best를 자동 복원하지 않습니다. 중단 후에는 새 런타임에서 01~05를 다시 실행한 다음, 중단된 난이도의 수집 또는 학습 셀을 다시 실행합니다.

- recovery 수집: 매 시도마다 `manifest.json`과 성공 episode를 동기화
- 학습: 매 1,000 iteration마다 `latest.pt` 동기화
- `latest.pt`: model, optimizer, AMP scaler, sampling RNG, CPU RNG, CUDA RNG 포함
- 설정·데이터·코드 signature가 다르면 잘못 이어 학습하지 않고 중단
- `training_complete.json`이 있으면 완성 학습을 다시 돌리지 않음

## Colab 실행 순서

1. T4 GPU 런타임을 선택합니다.
2. 01 CONFIG에서 `run_name`을 확인합니다. 중단 복구 시 이름을 바꾸지 않습니다.
3. 02에서 지정된 Git branch의 코드를 받습니다.
4. 03에서 GitHub 토큰을 입력하고 현재 run을 복원합니다.
5. 04에서 원래 프로젝트와 같은 pinned simulator dependency를 설치합니다.
6. 05에서 state dataset을 GitHub Release에서 받고 SHA256과 54/72/90차원 데이터를 검사합니다.
7. 필요한 난이도의 recovery 수집 셀을 실행합니다.
8. 바로 다음 학습 셀을 실행합니다. loss, 처리 속도, ETA, validation loss와 checkpoint 저장이 셀 출력에 표시됩니다.
9. 하나 이상의 난이도 학습이 latest.pt를 만들면 integrated candidate 셀부터 실행할 수 있습니다.
10. official smoke와 official default 평가로 simulator 성능을 확인합니다.
11. 더 안정적인 공개 seed 추정이 필요하면 선택 사항인 100-episode 셀을 실행합니다.
12. 마지막 셀에서 ZIP을 생성합니다.

## Loss와 실제 성능

학습 중 출력되는 loss는 다음 항을 합친 최적화 신호입니다.

- action imitation loss
- stage classification loss
- gate classification loss
- latent KL loss

이 값은 학습이 발산하는지 확인하는 데 사용하며 Kaggle 성능으로 해석하지 않습니다. 기본 candidate는 loss가 가장 낮은 `best_val.pt` 대신 현재 학습 상태인 `latest.pt`를 사용합니다.

실제 성능은 공식 `eval.py`가 simulator rollout 후 계산합니다.

```text
SORT ACCURACY = 올바른 bin에 들어간 parcel 수 / 전체 parcel 수
```

함께 출력되는 `mean_sorted`, `all_placed_rate`, `mis_sort_rate`도 공식 evaluator의 진단값입니다. official default는 4 episode라 분산이 큽니다. 선택 사항인 100-episode 셀도 같은 공식 `eval.py`와 같은 계산식을 사용하며, 공개 seed에서의 더 안정적인 추정치일 뿐 held-out Kaggle 점수는 아닙니다.
## 단일 inference 정책

```text
State observation
  → learned Stage ACT stage/gate supervisor를 매 step 실행
  → stage별 실행 horizon만큼 decoder chunk를 buffer에 저장
  → buffer action을 순서대로 실행
  → stage 전환 또는 recovery 승인 시 해당 env의 남은 buffer 폐기
  → 새 stage에서 즉시 replan
  → learned gripper logit을 짧게 안정화
```

- state dimension: Easy 54, Medium 72, Hard 90
- trained history/chunk: checkpoint의 설정 사용
- 실행 horizon: `pick=2, carry=6, place=2, done=1`
- gripper 변경: `abs(logit) >= 0.5`인 반대 명령이 2번 연속일 때 적용
- 공식 evaluator의 199-action batch 경계에서 history, stage, buffer, gripper latch 자동 초기화
- action: `(N,4)`, `[-1,1]`, `+1=open`, `-1=close`

`pick`과 `place` 시연에는 OPEN/CLOSE가 모두 있으므로 stage 이름으로 gripper를 강제하지 않습니다. gripper FSM은 모델이 학습한 logit만 안정화합니다.

## 결과 위치

- 현재 학습: `/content/moveboxes_runs/moveboxes_stage_chunk_deadline_v1`
- 기본 평가 checkpoint: `<run>/<level>/checkpoints/latest.pt` (`best_val.pt`는 loss 기반이라 자동 선택하지 않음)
- 공식 평가 로그·영상: `<run>/<level>/integrated_official_eval`
- candidate: `<run>/integrated_candidate.zip`
- 원격 복구: GitHub Release `run-moveboxes_stage_chunk_deadline_v1`

candidate에는 inference source, 현재 run checkpoint 사본, 새 policy sidecar, `submission.yaml`, SHA256 provenance manifest만 포함합니다. recovery expert와 training 코드는 제출 ZIP에 포함하지 않습니다.

## 로컬 검증 범위

- stage-aware buffer 순차 실행과 최신 history replan
- env별 stage transition 및 same-stage recovery invalidation
- learned gripper hysteresis
- manual reset 및 공식 fixed episode boundary reset
- 54/72/90차원 checkpoint load와 action shape/range
- notebook code cell compile
- Stage ACT 전체 테스트

로컬 Windows에는 ManiSkill/SAPIEN GPU 환경이 없으므로 실제 물리 rollout은 Colab의 integrated sanity와 official smoke 셀에서 처음 실행합니다.
