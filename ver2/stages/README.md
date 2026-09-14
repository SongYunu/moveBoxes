# 단계와 복구를 학습하는 State ACT

[Colab에서 열기](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/notebooks/moveboxes_stages_colab.ipynb)

기존 단계 v1의 첫 집기 실패는 [학습/실행 조건 수정](PRIOR_FIX.md)을 적용했습니다.
기존 모델이 있으면 [보정 노트북](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/notebooks/moveboxes_stage_repair_colab.ipynb)에서 성공 시연을 재사용할 수 있습니다.

Colab **2026.07 / Python 3.12 / T4**에서 01~05를 실행한 뒤 원하는 난이도의 셀을 순서대로 실행합니다.
Drive는 사용하지 않습니다. 03에서 기존 방식으로 GitHub 토큰을 입력합니다.

| 셀 | 내용 |
|---|---|
| 01~05 | 설정, 코드, 인증/복원, 설치, 데이터/GPU 점검 |
| 06~10 | Easy: 짧은 동작 확인 → 복구 시연 수집 → 학습 → 테스트/진단 → 최종 평가 |
| 11~15 | Medium: 같은 순서 |
| 16~20 | Hard: 같은 순서 |
| 21~22 | 결과 보기, 제출 모델 패키징 |

새 기본 `run_name`은 `moveboxes_stage_act_v2`입니다. 기존 ACT/DP 체크포인트는 구조가 달라 사용할 수 없습니다. 단계 v1 가중치는 보정 학습에 사용할 수 있습니다.
같은 이름으로 연결하면 GitHub Releases에서 수집 데이터·모델·점수·영상을 복원합니다.
학습은 마지막 업로드 체크포인트의 optimizer/RNG 상태부터, 수집과 평가는 완료된 에피소드 다음부터 재개합니다.
코드나 학습/수집 설정을 바꿔 다른 실험을 시작할 때에는 `run_name`을 바꿉니다.

## 변경한 방식

- 작은 ACT에 **현재 단계 예측 + 유지/완료/복구 예측**을 추가했습니다. 기본 크기는 Easy 1.28M, Medium 1.40M, Hard 1.53M입니다.
- 행동은 실행과 같은 zero-latent 경로를 직접 학습합니다. 미래 정답 행동을 입력으로 쓰는 posterior 경로는 기본 학습에서 제외합니다.
- 모델의 완료 또는 복구 확신이 설정값보다 높을 때 단계를 변경합니다. 실제 성공을 보장하는 판정기는 아니며, 분류가 틀리면 잘못 전환하거나 머무를 수 있습니다.
- 단계가 바뀌거나 복구가 승인되면 이전 XYZ 행동 예측을 무효화합니다. 모든 실제 행동은 학습된 decoder가 출력합니다.
- 집기·운반·놓기와 대상 상자가 바뀌는 경계에서 행동 chunk의 학습 마스크를 끊습니다.
- 이미 지나간 전환 시점을 놓쳤을 때도 회복하도록, 학습 샘플 25%에서는 이전 단계 메모리를 바꾸어 supervisor를 학습합니다. 관측과 행동 정답은 바꾸지 않습니다.
- 복구 시연 35%, 단계 전환 25%, 전체 구간 40%를 기본 표집합니다. 해당 풀이 비어 있으면 사용 가능한 풀로 대체합니다.

## 복구 시연

수집용 expert는 공식 scripted expert를 참고해 분리 구현했습니다. 제한된 이동 노이즈와 운반 중 집게 열기 교란을 넣고, 실제로 틀어진 상태에서 다시 계산한 expert 행동을 저장합니다.
**깨끗한 expert 정답과 교란된 실행 행동을 따로 저장**하며, 교란된 행동 자체를 정답으로 학습하지 않습니다.
수집 expert는 집기/놓기 확인 없이 시간 초과만으로 다음 상자로 넘어가지 않습니다. 같은 상자를 재시도하고, 재시도 한도를 넘으면 실패로 기록합니다.

기본은 난이도별 성공 시연 16회, 최대 시도 48회입니다. 공식 simulator `success_count`로 모든 상자의 정답 분류를 확인한 에피소드만 학습에 사용합니다.
수집은 Easy 500 / Medium 900 / Hard 1400 스텝까지 허용하지만 **모델 평가는 기존 200스텝 설정**을 유지합니다. 긴 수집 제한은 평가 제한을 늘리지 않습니다.
수집이 목표 횟수에 못 미치면 학습을 시작하지 않고 `LEVEL/collection/manifest.json`에 실패 이유를 남깁니다.

기존 데이터의 단계는 관측에서 얻은 약한 기하 라벨이며, 수집 데이터는 정확한 물체별 grasp 정보로 보완합니다. 로컬 모델의 실행에는 expert·환경 내부 상태가 사용되지 않습니다.
수집/라벨 코드와 expert는 최종 제출 ZIP에서 제외합니다.

## 평가와 검증 범위

테스트 8회, 별도 시드 진단 2회, 튜닝 8회, 최종 평가 100회입니다. 수집/테스트/튜닝/최종 시드는 서로 분리합니다.
`test_metrics.json`과 `metrics.json`에 기존 분류 점수와 집기 사이클 외에 단계별 체류, 전환, 복구 결정 횟수를 기록합니다.
진단 `traces/.../seed_*.json`에는 매 스텝 상태·행동·단계·판정 확률이 있습니다. 집기 사이클에는 같은 상자 재집기도 포함됩니다.

로컬 CPU에서 단계 정책·보정 경로 테스트와 세 난이도 실제 데이터 각 4개 시연의 전체 크기 모델 forward/backward를 확인했습니다.
단계 v1의 실제 Easy/Medium 시연 수집은 확인했지만 첫 집기 성능 퇴보가 있었습니다. **수정한 모델의 실제 물리 시뮬레이션과 점수 향상은 아직 검증하지 않았습니다.** 신규 학습은 06/11/16의 짧은 실행부터 확인하세요.
수집은 단순 noisy-expert augmentation이며 DAgger나 DART의 전체 알고리즘을 구현했다고 주장하지 않습니다.

참고: [공식 제출 규정](https://github.com/marso-robotics/berlin-marso-hackathon/blob/6048f33217f26ae39009a812f53c81171517f393/SUBMISSION.md),
[ACT](https://tonyzhaozh.github.io/aloha/), [DAgger](https://proceedings.mlr.press/v15/ross11a.html),
[DART](https://proceedings.mlr.press/v78/laskey17a.html).
