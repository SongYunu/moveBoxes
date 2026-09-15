# State Stage ACT 단일 deadline 정책

[Colab에서 열기](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/stage-act-chunk-compare/notebooks/moveboxes_stage_deadline_colab.ipynb)

## 적용한 로직

```text
State observation
  → 기존 learned Stage ACT의 stage/gate 판단
  → stage별 action chunk를 buffer에 저장
  → 순서대로 실행
  → stage 전환 또는 recovery 승인 시 남은 buffer 즉시 폐기
  → 새 stage에서 즉시 replan
  → learned gripper logit을 짧게 안정화
```

한 policy 안에서 모두 동작합니다. deadline Colab은 사용자가 지정한 현재 checkpoint만 사용합니다. 과거 EasyLab/MediumLab이 실제 Stage ACT였다는 검증 결과는 모델 방향의 근거로만 남기며, 해당 best checkpoint를 자동 복원하지 않습니다.

기존 `stage_policy.py`, `StageACT`, training loop와 원본 checkpoint는 수정하지 않습니다. 새 `stage_chunk_policy.py`와 checkpoint 사본·새 sidecar를 별도 candidate 폴더에 둡니다.

## 기본 실행 설정

- State: Easy/Medium/Hard = 54/72/90차원
- 모델 history: checkpoint에 저장된 값 사용. 검증 모델은 16
- prediction chunk: checkpoint에 저장된 값 사용. 검증 모델은 16
- 실행 horizon: `pick=2, carry=6, place=2, done=1`
- stage/gate threshold: 기존 sidecar 값 사용
- gripper: decoder가 학습한 logit만 사용
- gripper 변경: `abs(logit) >= 0.5`인 반대 명령이 2번 연속일 때 적용
- episode reset: 공식 evaluator의 199-action batch 경계에서 history, stage, buffer, gripper latch 자동 초기화
- action: `(N,4)`, `[-1,1]`, `+1=open`, `-1=close`

현재 4단계의 `pick`에는 접근·파지·lift가, `place`에는 bin 접근·하강·release가 함께 들어 있습니다. 원본 시연에서도 두 stage에 OPEN/CLOSE가 공존하므로 stage 이름으로 gripper를 강제하지 않습니다. 이는 규칙 기반 controller 제출을 금지한 공식 조건에도 맞지 않습니다.

## Colab 순서

1. T4 GPU 런타임을 선택합니다.
2. 01 셀의 `CHECKPOINTS`에 현재 사용할 Stage ACT `.pt` 경로를 입력합니다.
3. checkpoint 옆 `policy_config.json`을 사용할 경우 `USE_SIDECARS=True`를 유지합니다.
4. 02~04를 실행해 공식 simulator 버전, checkpoint family, state/action shape, buffer, reset, env step을 확인합니다.
5. 05는 공식 `eval.py`로 1 episode smoke를 실행합니다.
6. 06은 repository에 고정된 `conf/eval/default.yaml`을 공식 `eval.py`로 실행합니다.
7. 07은 동일한 단일 policy를 flat import 가능한 candidate ZIP으로 검증·압축합니다.

결과 로그와 영상은 `MyDrive/moveboxes_stage_deadline/<RUN_NAME>_official_eval`에 남습니다. 출력 수치를 별도로 파싱하거나 다시 계산하지 않습니다. 공식 평가기의 stdout 전체를 `official_eval.log`로 저장합니다.

candidate에는 다음만 들어갑니다.

- inference source: `stage_chunk_policy.py`, `stage_policy.py`, `stage_model.py`, `stage_schema.py`, `act_v2_model.py`
- 사용자가 지정한 checkpoint의 사본과 새 `policy_config.json`
- `submission.yaml`
- 원본 checkpoint SHA256과 코드 commit을 기록한 `manifest.json`

training, weak-label 생성, scripted expert, recovery 수집 코드는 candidate에 포함하지 않습니다.

## 검증 범위

- 세 historical Stage ACT checkpoint의 실제 다운로드, SHA256, strict model load 통과
- 기존 정책 옵션을 끈 A 경로는 기존 `StagePolicy` 출력과 bitwise 일치
- buffer 순차 실행, 부분 batch stage 전환, recovery, 불확실 gate, gripper 안정화, episode reset 테스트
- 전체 Stage test 23개 통과
- GitHub CI의 Stage ACT, Ver2 ACT, notebook/storage checks 통과

Windows에는 ManiSkill/SAPIEN이 없어 새 단일 로직의 실제 물리 episode는 실행하지 않았습니다. Colab 05의 official smoke가 최초 실제 물리 검증입니다.
