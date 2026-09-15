# 현재 State Stage ACT의 deadline 추론 비교

[현재 checkpoint용 A/B/C Colab 열기](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/stage-act-chunk-compare/notebooks/moveboxes_stage_deadline_colab.ipynb)

기본 Colab은 과거 best를 자동 복원하지 않습니다. 01 셀의 `CHECKPOINTS`에 현재 사용할 State Stage ACT 경로를 지정합니다. 과거 hash-pinned 모델은 `USE_KNOWN_BASELINES=True`로 명시한 경우에만 선택 기능으로 복원합니다.

## 결론과 기존 모델 확인

오늘 우선순위는 기존 State Stage ACT의 재현과 추론 비교입니다. 새 모델·새 stage 체계·재학습은 추가하지 않았습니다. **기존 `stage_policy.py`, 모델, 학습 루프, best 가중치를 그대로 유지합니다.**

| 확인한 과거 결과 | 체크포인트 | 학습 step | 실제 파일 SHA256 앞 12자리 |
|---|---|---:|---|
| Easy 100 episode, sort accuracy 100% | EasyLab `block_03.pt` | 6,000 | `0fbbef92aacc5` |
| Medium 8 episode, sort accuracy 40.625% | MediumLab `block_02.pt` | 4,000 | `f7c0d5edf5b2b` |
| Hard 8 episode, sort accuracy 4.167% | Hard Stage v1 `best_val.pt` | 18,000 | `96d30fe99cdf6` |

Easy/Medium 실제 PT의 SHA256이 각 점수 파일의 `protocol.checkpoint_sha256`과 일치합니다. 두 PT는 `moveboxes-stage-act-v1` 형식이며 `stage_embedding(4,128)`, `stage_head(4,128)`, `gate_head(3,128)` 가중치가 존재합니다. 현재 StageACT로 `strict=True` 로딩했을 때 누락·추가 키가 없습니다. 평가 당시 정책·평가기 소스 해시도 현재 기존 소스와 일치합니다. 따라서 성공 모델은 **EasyLab/MediumLab에서 보정한 Stage ACT**입니다. 초기 Stage ACT v1 Easy 0% 체크포인트와 혼동하면 안 됩니다.

위 결과는 과거 서로 다른 평가 조건/시드 기록입니다. 하나의 새 Overall 점수로 합산하지 않습니다. 세 PT 모두 공개 Release에서 실제 파일을 받아 기록된 SHA256과 `moveboxes-stage-act-v1` 구조를 재검증했습니다. Hard target v2/Object ACT/DP 가중치를 이 로더에 넣으면 명시적으로 거절합니다.

## 현재 구현을 읽은 결과

- 모델: `act_v2_model.StateACT`의 transformer를 상속한 `stage_model.StageACT`입니다. history 16, 예측 chunk 16, action 4입니다. observation의 상대 위치 feature를 train 통계로 정규화하고, 통계는 모델 buffer로 저장됩니다.
- Stage: `pick / carry / place / done`, gate: `hold / complete / recover`. 이전 stage embedding과 관측에서 새 stage/gate를 학습하여 예측하고, 승인한 stage embedding을 decoder query에 더합니다. stage는 제공된 state 벡터의 필드가 아닙니다.
- `pick`에는 접근·파지·lift가 포함됩니다. `place`에는 bin 접근·하강·release가 포함됩니다. 따라서 7단계용 gripper 고정 표를 그대로 사용할 수 없습니다.
- 학습: `stage_data.StageWindows`는 stage/target 경계와 교란 이후 action을 마스킹합니다. actions는 환경과 같은 `[-1,1]`로 clip하며 별도 action 표준화를 역변환하는 구조가 아닙니다. gripper는 BCE logit, xyz는 회귀입니다.
- 실행: 기존 정책도 과거 chunk의 해당 시점 XYZ를 최대 4개 혼합합니다. 매 step 재예측을 오류로 간주하지 않습니다. gripper는 최신 logit 부호를 사용합니다. stage 변경·동일 stage recovery 시 기존 예측을 generation 단위로 무효화합니다.
- Prior: 현재 학습 기본값은 배포와 같은 zero-latent 경로입니다. 과거 posterior-only 학습 문제는 이미 수정되어 있습니다.
- 저장: `best_val.pt`는 검증 loss 기준이고, 실제 성공 모델은 simulator 결과로 선택한 `block_XX.pt`입니다. `latest`나 `best_val`이라는 이름만으로 성공 모델을 고르면 안 됩니다.

### State 필드 (0부터 시작하는 반개구간)

공식 `warehouse_sort/env.py::_get_obs_extra`, Panda proprioception, 원본 HDF5와 대조했습니다. 상자 수 P는 Easy/Medium/Hard = 2/4/6입니다.

| 필드 | slice | 차원 |
|---|---|---:|
| 관절 위치, 마지막 두 관절은 finger | `[0:9]` | 9 |
| 관절 속도 | `[9:18]` | 9 |
| TCP xyz + quaternion | `[18:25]` | 7 |
| is_grasped | `[25:26]` | 1 |
| 상자별 xyz + quaternion | `[26:26+7P]` | 7P |
| 상자 색상 one-hot | `[26+7P:26+9P]` | 2P |
| bin 위치 | `[26+9P:32+9P]` | 6 |
| bin 색상 | `[32+9P:36+9P]` | 4 |

합계 **54/72/90**. RGB 트랙의 state는 앞 26차원입니다. state 트랙에서는 평가 시 전체 state를 제공합니다. [공식 제출 지침](https://github.com/marso-robotics/berlin-marso-hackathon/blob/6048f33217f26ae39009a812f53c81171517f393/SUBMISSION.md)은 state 트랙을 허용하고 rule-based controller 제출을 금지합니다. gripper convention은 공식 README와 Panda position controller의 finger target 범위에서 **+1=open, -1=close**입니다.

### 원본 600개 시연 stage 분포

현 `StageLabels`로 각 난이도 200개 전체를 다시 계산했습니다. 기하 기반 약한 라벨이며 환경의 실제 stage 정답이 아닙니다.

| stage | Easy | Medium | Hard |
|---|---:|---:|---:|
| pick | 15,000 | 31,071 | 54,550 |
| carry | 3,800 | 7,727 | 11,695 |
| place | 4,200 | 8,449 | 11,400 |
| done | 0 | 0 | 0 |

Easy `pick`에는 OPEN 8,200 / CLOSE 6,800, `place`에는 OPEN 600 / CLOSE 3,600 action이 공존합니다. **stage마다 gripper 값을 고정하면 시연과 직접 충돌합니다.** DONE은 원본 시연의 action 시점 라벨에서 없으며 성공 복구 시연 등에서 보완됩니다. 기존 학습에는 전환/복구 oversampling과 첫 집기 표집이 이미 있어, 이번에는 sampling/retraining을 추가하지 않았습니다.

## A/B/C

| 실험 | 실행 | gripper |
|---|---|---|
| A | 기존 temporal ensemble, 매 step decode | 기존 최신 logit 부호 |
| B | `pick=2, carry=6, place=2, done=1`개 순차 실행 | 실행 중인 chunk의 학습 logit 부호 |
| C | B와 동일 | 학습 logit confidence + 연속 2 step 확인 필터 |

`stage_chunk_policy.ChunkStagePolicy`는 supervisor를 **매 step** 호출하고 필요한 환경만 다시 decode합니다. stage 전환 또는 같은 stage recovery가 승인되면 해당 환경의 buffer를 버리고 즉시 새 action을 실행합니다. episode reset은 history/buffer/gripper filter를 모두 비웁니다. threshold가 낮은 미승인 전환은 기존 stage를 유지합니다.

실행 길이는 새로운 architecture horizon이 아닙니다. checkpoint의 history와 prediction chunk는 그대로 읽고, 실행 길이가 학습 chunk보다 길면 오류를 냅니다. B/C에서는 temporal ensemble을 사용하지 않습니다. 학습 중 chunk 끝이 stage 경계로 마스킹되므로 긴 open-loop 실행이 무조건 좋아지는 것은 아닙니다. 기본값은 재학습 없는 보수적인 비교 시작점입니다.

C의 `gripper_fsm`은 **stage별 고정 OPEN/CLOSE FSM이 아닙니다.** 학습 logit만 따르며 `abs(logit)>=0.5`가 연속 두 action에서 반대 부호를 지시할 때 전환합니다. stage/recovery 때 필터를 초기화하여 새 명령을 즉시 적용합니다. 놓기 지연 가능성이 있으므로 실험용이고 기존 제출 정책은 A입니다. C의 제출 적합성/점수 우위는 확인하지 않았습니다.

## 실행

새 Colab의 01~06을 순서대로 실행합니다. 기존 프로젝트와 같은 simulator 버전을 사용하고 이미 설치된 패키지는 업그레이드하지 않습니다. 공개 GitHub Release의 URL·파일 크기·SHA256을 코드에 고정했기 때문에 GitHub 토큰 없이 정확한 가중치만 별도 디렉터리로 다운로드합니다. 최신 모델로 임의 대체하지 않습니다. 결과는 기본적으로 Google Drive의 `MyDrive/moveboxes_stage_compare/abc_8seeds_v1`에 episode마다 저장됩니다.

처음에는 `SEEDS=[61000]`과 별도 `OUTPUT`으로 1 episode smoke를 실행할 수 있습니다. 이후 8개 seed와 새 output으로 A/B/C를 비교하세요. 이번 작업에서는 Colab 연결이 없어 실제 simulator episode/점수를 생성하지 않았습니다.

CLI는 설치된 공식 starter의 `conf/`가 있는 디렉터리에서 실행합니다:

```bash
python /content/moveBoxes_stage_compare/ver2/stages/stage_compare.py /content/stage_compare_abc_v1/config.json
```

로컬 가중치와 HDF5만 점검하려면 `--audit-only --device cpu`를 추가합니다. config의 난이도별 spec에 `data` HDF5 경로가 있으면 stage 분포와 실제 시연 관측 replay까지 검사합니다. `data`가 없으면 synthetic shape 검사라고 명시합니다.

직접 연결할 때에는 `load_chunk_stage(..., stage_aware_chunk=True, stage_horizons={...}, gripper_fsm=False)`를 사용합니다. 기본 `load_stage`나 기존 notebook은 바뀌지 않습니다. baseline policy config의 ensemble/gate threshold를 반드시 함께 사용하세요.

### 평가/배포 reset

고정한 upstream `rollout_metrics`는 여러 episode 사이에 policy.reset을 호출하지 않습니다. 기존 전용 Stage evaluator는 각 seed 전에 이미 reset하고 있으므로 이 문제가 기존 점수의 원인이라는 증거는 없습니다. 새 비교기도 **한 episode씩 공식 평가 함수를 호출하며 매번 명시적으로 reset**합니다. 공식 action budget인 `max_steps-1` 실행과 metric을 그대로 사용합니다.

별도 judge/배포에서 여러 episode에 같은 policy 객체를 재사용한다면 env.reset 이후 policy.reset도 호출해야 합니다. `.act(obs)`만으로 같은 batch 크기의 임의 환경 reset을 정확히 감지할 수 없습니다. 이 저장소는 외부 judge 코드를 바꾸거나 관측으로 reset을 추측하지 않습니다.

### 출력

- `audit.json`: checkpoint hash, architecture, state 필드, action/reset 검사, 선택적 stage 분포.
- `<level>/A.json`, `B.json`, `C.json`: 동일 seed별 결과와 정책 설정, 소스 hash, 환경 설정. 완료 episode마다 저장하고 동일 조건에서 재개합니다. 조건이 달라지면 덮어쓰지 않고 새 output을 요구합니다.
- `comparison.json`, `comparison.md`: Easy/Medium/Hard/Overall 표. 미실행 난이도는 N/A, 세 난이도를 모두 완료했을 때만 `0.2/0.3/0.5` Overall을 계산합니다.
- episode별 실제 sort/mis-sort metric, 파지 관측 여부, gripper 반전, stage 체류, decode 호출 수, 시간 제한까지 미완료 여부를 기록합니다. carry 중 grasp 소실은 `possible_carry_drops`라는 진단용 추정치이며 실제 drop 정답으로 취급하지 않습니다.

새 결과가 기존 A보다 좋은지 확인하기 전에는 best/sidecar를 교체하지 마세요. deadline Colab은 선택 결과를 원본과 분리된 `_candidate` 폴더에 복사하고, 제출용 flat module import와 실제 checkpoint action을 다시 검사한 뒤 ZIP을 만듭니다. 비교 seed로 고른 정책은 별도 최종 seed에서 평가해야 합니다.

## 로컬 검증

원본 Easy/Medium 실제 best checkpoint load·40 frame observation replay·action shape/range·episode reset이 A/B/C 모두 통과했습니다. A는 기존 정책과 bitwise 동일한 출력입니다. 버퍼 순차 실행, 부분 batch 전환, 동일 stage recovery, 불확실한 gate, 그리퍼 필터, episode reset, 잘못된 checkpoint/horizon 거절, 평가 재개, 읽기 전용 복원과 notebook 컴파일을 테스트합니다. 세 state dimension은 합성 StageACT checkpoint로도 점검합니다.

```bash
python -m unittest discover -s tests_stages -v
```

세 historical checkpoint의 실제 다운로드·SHA256·모델 로딩·A/B/C synthetic action/reset 검사는 통과했습니다. 새 B/C의 실제 시뮬레이션 성능은 아직 검증하지 않았습니다. Windows 개발 환경에는 ManiSkill/SAPIEN이 없어 Colab에서 확인해야 합니다.
