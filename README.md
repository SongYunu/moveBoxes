# moveBoxes

**State Stage ACT 단일 deadline 정책:** [Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/stage-act-chunk-compare/notebooks/moveboxes_stage_deadline_colab.ipynb) · [구성과 실행 안내](ver2/stages/DEADLINE_POLICY.md).
GitHub 토큰 인증과 state dataset 검증부터 State Stage ACT 학습, 중단 복구, stage-aware chunk·learned gripper 안정화, 공식 `eval.py` 평가, rollout 영상의 Colab inline 재생, ZIP 생성까지 한 Colab에서 실행합니다. Drive는 사용하지 않으며 과거 best는 자동 복원하지 않습니다. **새 로직의 실제 Colab 점수는 아직 측정하지 않았습니다.**

**공개 사전학습 모델을 쓰는 RGB 공동 학습:** [SmolVLA + Octo 통합 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/foundation/notebooks/moveboxes_foundation_both_colab.ipynb) · [설명](foundation/README.md).
각 모델은 Easy·Medium·Hard RGB 시연 600개를 처음부터 함께 보며, 같은 최고 adapter 하나를 세 난이도에서 평가합니다. 모델과 시뮬레이터 의존성은 서로 다른 Python 환경에 격리했습니다.

**새 공동 학습 모델:** [Object Attention Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/unified/notebooks/moveboxes_joint_scratch_colab.ipynb) · [실행 안내](unified/JOINT.md).
Easy·Medium·Hard state 시연 전체를 처음부터 섞어, 모든 상자를 attention으로 보는 모델 하나를 학습합니다.
약 117만 파라미터, T4용 배치 32. **01~06 준비 → 07 공동 학습 → 08~10 난이도별 테스트**입니다.
원본 데이터 CPU 학습·로딩 테스트는 통과했으며 새 모델의 실제 T4 성공률은 아직 측정하지 않았습니다. 아래 버전들은 이전 실험 기록입니다.

**기존 Medium 40.6% 모델을 Easy에 평가:** [학습 없는 별도 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/unified/notebooks/moveboxes_medium_block02_easy_test.ipynb) · [설명](unified/MEDIUM_PROBE.md).
원본 `block_02.pt`를 그대로 로드하고 Easy의 없는 상자 입력만 패딩합니다. 01~05 준비 → 06 Easy 평가.

**이전 Medium 집중 학습: [Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/unified/notebooks/moveboxes_unified_medium_colab.ipynb)** · [설명](unified/MEDIUM.md).
v2에서 Easy 100%를 낸 실제 PT를 그대로 가져와 Medium 20,000회 학습합니다.
기존 부분 성공 시연 32개와 원본 데이터를 재사용하며, Easy 성능 저하로 Medium 학습을 중단하지 않습니다.
01~06 준비 → 07 학습 → 08 테스트. 기존 Easy 원본과 최고 Medium 모델을 따로 보관합니다.

**이전 공유 모델 v2: [Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/unified/notebooks/moveboxes_unified_v2_colab.ipynb)** · [실행 순서·수정 내용](unified/V2.md).
기존 최종 Easy 100/100 성공 체크포인트에서 시작해 같은 모델을 Medium → Hard까지 이어 학습합니다.
실패 관련 전문가 시간 구간을 집중 표집하고 이전 데이터를 재사용합니다. 학습 모델은 구간 사이에
optimizer와 난수 상태까지 이어가며, 최고 성적 모델은 따로 보존합니다. 01~06 준비 → 07~08 Easy 확인부터 실행하세요.
원본/변환 Easy를 같은 시드에서 먼저 비교합니다. 새 버전의 GPU 성능은 아직 측정하지 않았습니다.
Medium 초기값을 Easy에 바로 적용해 실패했던 [통합 v1](unified/README.md)과 아래 개별 실험은 기록으로 유지합니다.

난이도별 코드는 [`easy/`](easy/README.md), [`medium/`](medium/README.md), [`hard/`](hard/README.md)에 분리되어 있습니다.
공통 모델과 GitHub 저장 코드는 루트 및 `ver2/`에 있습니다.

**Hard 실행:** [Hard 전용 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/hard/notebooks/moveboxes_hard_colab.ipynb) · [실행 안내](hard/README.md).
집기 실패 뒤 복구 시연 수집부터 Hard 학습·테스트·최종 평가까지 별도 Release에서 실행합니다.

**이전 Hard v3 실험:** [Easy/Medium 전이 재학습 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/hard/notebooks/moveboxes_hard_v3_colab.ipynb).
Easy와 Medium 최고 모델을 Hard 입력으로 각각 변환해 먼저 비교하고, 더 나은 시작점 하나만 기존 Hard 성공 시연으로 짧게 보정합니다.

**Medium만 베이스로 쓸 때:** [Hard v3.1 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/hard/notebooks/moveboxes_hard_v31_colab.ipynb).
Easy는 읽지 않으며, 06 셀에서 재학습 전 Medium 직접 전이를 먼저 시험합니다. 07 보정 학습은 선택 사항입니다.

**Medium v2.2:** [집기 높이 보정 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/medium/notebooks/moveboxes_medium_v22_colab.ipynb) · [실행 안내](medium/docs/V22.md).
v2.1의 모델과 32개 성공 시연을 재사용하고, 단계 판정을 고정한 채 집기 Z축과 그리퍼 타이밍만 짧게 보정합니다.

**Medium v2.1:** [부분 성공 학습 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/medium/notebooks/moveboxes_medium_v21_colab.ipynb) · [실행 안내](medium/docs/V21.md).
3개 분류 시연도 학습하고, 네 상자 완주 시연에만 속도 가중치를 줍니다. 시연 32개·최대 12,000회 보정입니다.

**Medium v2:** [별도 Colab 노트북](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/medium/notebooks/moveboxes_medium_v2_colab.ipynb) · [실행 안내](medium/docs/V2.md).
199행동 내 성공 시연을 먼저 시험하고, 기존 최고 모델을 최대 3,000회 보정합니다. 결과는 별도 Release에 저장합니다.

**Medium 실행:** [Medium 전용 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/medium/notebooks/moveboxes_medium_colab.ipynb) · [실행 안내](medium/README.md).
Easy 집중 실험과 같은 학습 방식으로 Medium의 네 상자를 학습합니다. 01~05 준비 → 06 반복 학습 → 07 테스트 순서입니다.

**Easy에 집중할 때:** [Easy 전용 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/easy/notebooks/moveboxes_easy_colab.ipynb) · [실행 안내](easy/README.md).
2,000회마다 실제 평가하고 가장 좋은 모델을 보존합니다. Medium/Hard 학습 셀은 없습니다.

**GitHub 422 저장 오류:** 한 Release의 파일 수 한도에 도달한 경우 `-part-002` 등으로 자동 이어 저장합니다.
현재 런타임은 초기화하지 않고 02 코드 로드 → 03 연결 셀을 다시 실행하세요. 기존 모델과 snapshot은 보존합니다.

**단계 v1에서 첫 집기 실패가 발생한 경우:** [보정용 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/notebooks/moveboxes_stage_repair_colab.ipynb)의 01~06을 실행하세요.
기존 모델·성공 시연을 재사용해 실행 조건으로 2,000회 보정 학습하고 테스트합니다. [원인과 검증 기록](ver2/stages/PRIOR_FIX.md).

**새 단계 ACT 버전:** [Colab에서 열기](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/notebooks/moveboxes_stages_colab.ipynb) · [실행 순서와 변경 사항](ver2/stages/README.md).
집기·운반·놓기의 완료/복구 판단을 학습하고, 복구 시연 수집·학습·테스트를 난이도별로 실행합니다.
기존 DP와 ACT ver2 파일 및 결과는 그대로 유지합니다.

WarehouseSort 상자 분류를 위한 **Colab T4용 State Diffusion Policy** 실험입니다.
Easy / Medium / Hard 각각 학습·테스트·평가 셀이 있으며, 다음 상자 집기 구간을 강조해 학습합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/notebooks/moveboxes_colab.ipynb)

## 시작하기

1. 위 **Open in Colab** 버튼으로 노트북을 엽니다.
2. Colab **런타임 2026.07 / Python 3.12 / T4 GPU**를 선택합니다.
3. GitHub에서 이 저장소만 선택한 fine-grained token을 만듭니다.
   저장소 권한은 **Contents: Read and write**가 필요합니다.
4. 03 셀은 기존 **`GH_TOKEN` 환경변수 → Colab 보안 비밀 → 비공개 입력창** 순서로 토큰을 받습니다.
   보안 비밀 등록 없이 입력창에 붙여넣어도 됩니다. 입력값은 현재 런타임에만 남습니다.
   개인 사본에 직접 지정하려면 03 셀의 예시 한 줄을 사용하세요. 토큰이 들어간 사본은 공개 Git에 올리지 않습니다.
5. 01~05 셀을 실행한 뒤 원하는 난이도의 셀을 순서대로 실행합니다.

| 셀 | 내용 |
|---|---|
| 01~05 | CONFIG → GitHub 코드 로드 → 인증/복원 → 설치 → 데이터/GPU 확인 |
| 06~09 | Easy: 짧은 동작 확인 → 전체 학습 → 8회 테스트+영상+별도 2회 행동 기록 → 튜닝/최종 100회 평가 |
| 10~13 | Medium: 같은 순서 |
| 14~17 | Hard: 같은 순서 |
| 18~19 | 결과/영상 보기 → 모델 패키징 |

노트북은 코드 셀만 19개이며 별도 파일 업로드나 Drive 마운트가 필요하지 않습니다.
코드는 이 저장소에서, 데이터는 [dataset-v1 Release](https://github.com/SongYunu/moveBoxes/releases/tag/dataset-v1)에서 받습니다.
공개 데이터 다운로드 자체는 토큰이 필요하지 않으며 결과 백업/복원에 토큰을 사용합니다.

**기존 v02 실행 복구:** 같은 `run_name`을 유지하고 새 노트북의 01~05 셀을 실행하세요.
저장된 Easy 모델이 있으면 07 셀의 재학습은 생략됩니다. 08 셀은 테스트와 행동 기록,
09 셀은 최종 평가 재개입니다. 기존 평가 코드와 정책 계산은 유지했으므로, 동일한 모델·설정으로
74/100회가 저장된 경우 남은 26회만 실행합니다. Medium 학습은 11, Hard 학습은 15 셀입니다.
02 셀을 다시 실행해 연결 상태가 초기화되어도 다음 작업에서 자동으로 연결합니다.

## 데이터

| 파일 | 내용 |
|---|---|
| `marso_state_data.zip` | 학습에 사용하는 Easy/Medium/Hard state HDF5와 JSON |
| `marso_rgb_data.zip` | RGB 트랙 원본 HDF5와 JSON; 이 노트북에서는 사용하지 않음 |
| `data/manifest.json` | 다운로드 주소, 크기, SHA-256, 출처 |

현재 state 모델에는 **state ZIP만 자동 다운로드**합니다. 데이터 복사본은 원본 이용조건을 따르며
원본 데이터에 새 라이선스를 부여하지 않습니다.

출처: [Kaggle competition](https://www.kaggle.com/competitions/marso-hack-berlin-2026-robot-parcel-sorting-challenge),
[MARSO 공식 코드](https://github.com/marso-robotics/berlin-marso-hackathon).
환경 코드는 `6048f33217f26ae39009a812f53c81171517f393` 커밋에 고정합니다.

## 결과 저장과 복원

기본 결과 Release는 [run-moveboxes_next_pick_v02_benchmark](https://github.com/SongYunu/moveBoxes/releases/tag/run-moveboxes_next_pick_v02_benchmark)입니다.
**처음 Colab 실행 시 생성**되며, 아직 실행하지 않았다면 링크가 없을 수 있습니다.

- 학습 체크포인트를 파일에 완전히 저장한 뒤 GitHub에 업로드합니다.
- 평가 에피소드 하나가 끝날 때 점수와 진행 상태를 업로드합니다.
- 같은 평가 프로세스에서는 바뀌지 않은 모델 파일의 복사·해시 계산과 Release 목록 재조회를 생략합니다.
- 각 난이도 학습·테스트·평가 완료 시 로그, 설정, 영상도 업로드합니다.
- 공통 결과 백업 전에 요약을 갱신합니다. 중간 점수와 `74/100회` 같은 진행 상태는 최종 점수와 구분합니다.
- 필요한 파일을 모두 업로드한 후에 복원용 snapshot을 게시합니다. 업로드 중 끊기면 이전 snapshot을 사용합니다.
- 새 런타임에서 같은 `run_name`으로 01~03 셀을 실행하면 결과를 복원합니다.
  평가 재개는 04~05 셀까지 준비한 후 해당 평가 셀을 실행합니다.
- 현재 로컬 결과가 있으면 원격 결과로 덮어쓰지 않습니다. 별도 실험에는 새 `run_name`을 사용하세요.

Release 파일명은 내용 해시를 포함합니다. 원래 난이도별 폴더 구조는 snapshot JSON을 통해 복원합니다.
소스 Git에는 모델·영상·데이터 바이너리를 커밋하지 않습니다.
이 저장소는 **공개 저장소**이므로 올린 모델·로그·영상도 공개됩니다.

**복구 범위:** 마지막으로 업로드까지 완료한 체크포인트/에피소드까지 복구됩니다.
공식 trainer는 모델과 EMA 가중치만 저장합니다. optimizer/scheduler를 포함한 **정확한 이어 학습은 아직 지원하지 않습니다.**
중단 뒤 기존 체크포인트가 있으면 재학습을 생략하고 그 모델을 평가합니다.
100 iteration 동작 확인용 모델은 작은 로컬 `checks/`에만 저장합니다.
업로드는 추가 시간이 들며 GitHub 토큰 권한과 네트워크 연결이 필요합니다.
새 학습에서는 마지막 iteration의 모델을 최종 학습 평가 전에 별도 저장합니다
(예: Easy `30000.pt`). 기존 학습에서 저장되지 않은 마지막 가중치를 소급 복원하지는 않습니다.
학습 로그에는 매 iteration의 진행 막대 대신 `log_freq` 간격의 진행률과 loss를 남깁니다.

## 모델과 평가

- T4 기본: batch 32, UNet `[32, 64, 128]`, 관측 길이 2, 예측 길이 16.
- 두 번째 이후 안정적인 집기 사이클 주변 시연을 3배 가중 표집합니다.
  같은 상자 재집기도 포함될 수 있으며 탐지 내용은 `sampling_audit.json`에 기록합니다.
- 기본 4스텝마다 행동을 다시 생성하고 튜닝 시 4/8스텝을 비교합니다.
- Easy 30,000 / Medium 50,000 / Hard 60,000 iteration.
- 8회 빠른 테스트, 별도 시드로 튜닝, 최종 100회 공개 설정 자체 평가.
- 평가 제한은 기존 200스텝 설정을 유지하며 공식 rollout의 `max_steps - 1` 동작을 따릅니다.
- 가중 점수는 `0.2 × easy + 0.3 × medium + 0.5 × hard`입니다.
  실제 제출 점수와 자체 평가를 구분해야 합니다.
- 테스트에는 1개/2개 이상 정답 분류 비율, 집기 사이클, 집게 명령 전환, 영상이 포함됩니다.
- 테스트 셀의 `experiment.diagnose(level)`은 테스트 시드 2개로 별도 실행하고,
  `LEVEL/traces/실행ID/seed_40000.json` 등에 매 스텝의 전체 state와 4차원 action을 저장합니다.
  GitHub snapshot으로 함께 복원되며, 같은 모델·설정의 기록은 재사용합니다.
  진단은 정책 행동을 수정하지 않고 최종 점수에도 포함하지 않습니다.

기존 Easy 실행에서는 T4 학습 완료와 최종 평가 74회 저장을 확인했지만,
두 번째 상자 집기 실패가 남았습니다. 이번 수정은 실행·복구·기록 개선이며 성능 향상을 검증한 모델 변경은 아닙니다.
새 GPU 실행은 Colab에서 확인해야 합니다. 첫 실행에는 각 난이도의 짧은 동작 확인 셀을 사용하세요.

## 로컬 개발

```bash
git clone https://github.com/SongYunu/moveBoxes.git
cd moveBoxes
python build_github_notebook.py
python -m unittest discover -s tests -v
```

`marso_github.py`는 GitHub 저장/복원을, `github_store.py`는 Release snapshot을,
`github_data.py`는 데이터 다운로드와 해시 검증을 담당합니다.
이전 학습·정책 모듈은 알고리즘 비교를 위해 유지합니다.
코드를 수정한 뒤 `build_github_notebook.py`를 실행하면 진입 노트북을 다시 생성할 수 있습니다.

토큰 발급 안내: [GitHub 공식 문서](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
