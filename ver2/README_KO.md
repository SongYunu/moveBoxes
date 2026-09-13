# moveBoxes ver2: 작은 State ACT

[Colab에서 ver2 열기](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/notebooks/moveboxes_ver2_colab.ipynb)

기존 Diffusion Policy 코드와 `notebooks/moveboxes_colab.ipynb`는 그대로 둡니다.
새 코드와 결과는 별도이며, 기존 DP 가중치를 ACT에 불러오지 않습니다.

## 실행

Colab 2026.07 / Python 3.12 / T4 GPU에서 새 노트북을 엽니다.
01~05 셀을 순서대로 실행하고, 03 셀에 토큰을 입력합니다. 기존 GH_TOKEN도 사용할 수 있습니다.

| 난이도 | 100회 학습 동작 확인 | 전체 학습 | 8회 테스트·영상·2회 행동 기록 | 튜닝·최종 100회 |
|---|---:|---:|---:|---:|
| Easy | 06 | 07 | 08 | 09 |
| Medium | 10 | 11 | 12 | 13 |
| Hard | 14 | 15 | 16 | 17 |

18 셀에서 결과를 보고, 19 셀에서 제출용 모델 묶음을 만듭니다.
기본 실행 이름은 `moveboxes_act_ver2`, 결과 Release는 `run-moveboxes_act_ver2_benchmark`입니다.
데이터는 기존 dataset-v1 Release를 재사용합니다. 별도 Drive 조정은 필요 없습니다.
런타임이 끊기면 같은 설정으로 01~05를 실행하고 해당 학습·평가 셀을 다시 실행하세요.

## 왜 바꿨나

2026-09-14에 읽은 기존 결과는 Easy 최종 100회 48%, Medium 빠른 테스트 8회 18.75%였습니다.
두 점수는 평가 시드와 횟수가 달라 직접 비교할 지표가 아닙니다.
Easy 행동 기록 2회에서는 첫 집기 뒤 이동이 지연되고, 분류 후에는 두 번째 상자 근처로 제대로 접근하지 못했습니다.
이것만으로 모든 실패 원인이 밝혀진 것은 아닙니다.

로컬 원본 state 시연 200개씩을 확인하면 XYZ 행동 성분 중 약 10.14% / 11.58% / 11.62%가
`[-1,1]` 범위를 벗어납니다. 기존 DP trainer는 이 값을 그대로 noise 예측 학습에 사용하고,
배포 정책은 `[-1,1]`로 자릅니다. ver2에서는 환경 실행 범위에 맞춰 정답도 자릅니다.
ManiSkill의 정규화 action 변환은 clip 후 scale을 수행합니다.
[공식 함수](https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/utils/gym_utils.html#clip_and_scale_action)

Hard의 집게 명령 77,645개 중 16개에는 중간값도 있습니다. 이를 버리지 않고
`(clip(g,-1,1)+1)/2`의 soft label로 학습합니다. 실행 시에는 최신 예측에서 열기/닫기를 선택합니다.

## 조합한 방법과 차이

- **ACT 아이디어:** 관측에서 미래 16개 행동을 한 번에 예측하는 Transformer와 CVAE 학습을 사용합니다.
  영상 encoder 대신 state 입력, 작은 MLP posterior, XYZ 회귀와 집게 분류를 쓰는 별도 구현입니다.
  원 논문의 모델이나 대회 우승 제출을 그대로 재현한 것은 아닙니다.
  [ACT 저자 프로젝트](https://tonyzhaozh.github.io/aloha/)
- **짧은 피드백 간격:** 매 스텝 새 관측으로 예측합니다. 과거 최대 4개 예측 중 현재 시점의 XYZ를 평균하고,
  최신 예측에 더 큰 가중치 `exp(-0.25 × age)`를 줍니다. 집게는 평균하지 않습니다.
  원 ACT 코드의 평균 가중치와는 구분되는 반응성 중심의 선택입니다.
  검증 시드에서는 평균 없음(1개)과 4개 평균을 비교합니다.
  [ACT 코드](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)
- **입력:** 최근 16개 state, 각 상자와 TCP 사이의 상대 위치, 색에 맞는 목표 함과 상자 사이의 상대 위치를 사용합니다.
  정규화 통계는 학습 시연에서만 계산합니다. 정적인 집기 대기 구간을 짧은 관측보다 구분하기 위한 설계입니다.
- **학습 정답:** 실행 범위로 clip하고, 시연 끝을 넘어가는 미래 행동에는 loss를 주지 않습니다.
  가짜 정지 행동을 반복해서 학습시키지 않습니다. XYZ는 Smooth L1, 집게는 BCE로 학습합니다.
- **작은 입력 증강:** 위치 좌표에 표준편차 1mm 노이즈를 추가하며 색·grasp bit는 그대로 둡니다.
  이것은 관측 증강입니다. 전문가에게 실패 상태를 재라벨링받는 DAgger나 DART 구현이 아닙니다.
  [DAgger](https://proceedings.mlr.press/v15/ross11a.html)

실패 상태에 대한 전문가 복구 시연을 추가하지 않았으므로 분포 변화나 연쇄 오류를 완전히 해결했다고 주장할 수 없습니다.
성능 향상 여부는 Easy 테스트/영상부터 확인해야 합니다. 기존 DP와 비교할 때는 같은 200스텝 제한을 유지합니다.
Medium·Hard 시연은 200스텝보다 긴 경우가 많아 시간 제한의 영향도 남아 있습니다.

## T4 자원과 검증

width 128, 2개 encoder/decoder 층, history 16, action chunk 16, latent 16, batch 64입니다.
전체 파라미터 수는 Easy 1,240,612 / Medium 1,367,332 / Hard 1,494,052입니다.
이미지 backbone과 반복 denoising이 없고 학습 중 시뮬레이터를 만들지 않습니다.
데이터는 CPU 메모리에 두고 모델·현재 batch만 GPU에 올립니다. CUDA 학습은 FP16 AMP를 사용합니다.
기본 학습량은 Easy 12,000 / Medium 20,000 / Hard 30,000 iteration입니다.
총 시간은 첫 실행의 실제 it/s와 ETA 출력으로 확인하세요. CPU 검사 시간은 T4 시간 추정치가 아닙니다.

시연을 trajectory 단위로 90/10 분리합니다. CVAE posterior 학습 loss와 별도로,
실제 배포처럼 latent=0일 때의 검증 loss로 `best_val.pt`를 고릅니다.
그다음 별도 튜닝 seed에서 best_val/latest와 평균화 설정을 비교합니다.
최종 100회 seed는 튜닝·빠른 테스트 seed와 겹치지 않습니다.

`latest.pt`에는 모델·정규화 통계·optimizer·AMP scaler·학습 단계·샘플 RNG·Torch RNG를 저장합니다.
저장된 모델/데이터/학습 코드와 설정이 같을 때만 이어 학습합니다.
CPU에서 중간 중단 후 재개한 결과와 연속 학습의 가중치가 동일함을 검사했습니다.
GPU 모델/라이브러리의 변경이나 CUDA 비결정성까지 동일한 수치를 보장하지는 않습니다.
2,000 iteration마다와 학습 종료 때 백업합니다. 강제 해제 시 마지막 완료 백업 이후 작업은 다시 계산합니다.

로컬 검증은 CPU의 텐서·학습·체크포인트 검사입니다. T4 물리 시뮬레이션 성공률은 아직 측정하지 않았습니다.
실제 시연에서 난이도별 20개 trajectory를 읽어 100 iteration씩 학습했으며,
50→100 iteration의 prior 검증 loss는 Easy 0.357→0.283, Medium 0.371→0.304,
Hard 0.486→0.358이었습니다. 이는 학습 코드의 동작과 손실 감소 확인이며 sorting 성공률이 아닙니다.

## 개발 검사

```bash
pip install torch numpy h5py
python -m unittest discover -s tests_ver2 -v
python build_act_v2_notebook.py
```

각 파일은 `ver2/`에 있고, Colab 설치 단계에서 별도 실행 프로세스가 import할 위치로 복사합니다.
기존 학습기·노트북·Release를 삭제하거나 덮어쓰지 않습니다.
