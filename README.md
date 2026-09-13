# moveBoxes

WarehouseSort 상자 분류를 위한 **Colab T4용 State Diffusion Policy** 실험입니다.
Easy / Medium / Hard 각각 학습·테스트·평가 셀이 있으며, 다음 상자 집기 구간을 강조해 학습합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/notebooks/moveboxes_colab.ipynb)

## 시작하기

1. 위 **Open in Colab** 버튼으로 노트북을 엽니다.
2. Colab **런타임 2026.07 / Python 3.12 / T4 GPU**를 선택합니다.
3. GitHub에서 이 저장소만 선택한 fine-grained token을 만듭니다.
   저장소 권한은 **Contents: Read and write**가 필요합니다.
4. Colab 왼쪽 열쇠 아이콘의 보안 비밀에 **`GH_TOKEN`**으로 등록하고 노트북 액세스를 허용합니다.
   토큰을 CONFIG, 코드, 출력, Git 커밋에 넣지 않습니다.
5. 01~05 셀을 실행한 뒤 원하는 난이도의 셀을 순서대로 실행합니다.

| 셀 | 내용 |
|---|---|
| 01~05 | CONFIG → GitHub 코드 로드 → 인증/복원 → 설치 → 데이터/GPU 확인 |
| 06~09 | Easy: 짧은 동작 확인 → 전체 학습 → 8회 테스트+영상 → 튜닝/최종 100회 평가 |
| 10~13 | Medium: 같은 순서 |
| 14~17 | Hard: 같은 순서 |
| 18~19 | 결과/영상 보기 → 모델 패키징 |

노트북은 코드 셀만 19개이며 별도 파일 업로드나 Drive 마운트가 필요하지 않습니다.
코드는 이 저장소에서, 데이터는 [dataset-v1 Release](https://github.com/SongYunu/moveBoxes/releases/tag/dataset-v1)에서 받습니다.
공개 데이터 다운로드 자체는 토큰이 필요하지 않으며 결과 백업/복원에 토큰을 사용합니다.

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
- 각 난이도 학습·테스트·평가 완료 시 로그, 설정, 영상도 업로드합니다.
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

실제 T4 학습이나 향상된 점수는 이 저장소 준비 과정에서 검증하지 않았습니다.
먼저 각 난이도의 짧은 동작 확인 셀을 통과시킨 뒤 전체 학습을 실행하세요.

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
