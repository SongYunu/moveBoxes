# Hard 전용 실행

[Hard 전용 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/hard/notebooks/moveboxes_hard_colab.ipynb)을
Colab 2026.07 / Python 3.12 / T4에서 실행합니다.

현재 권장 실험은 [Hard v3 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/hard/notebooks/moveboxes_hard_v3_colab.ipynb)입니다.
검증된 Easy와 Medium StageACT를 각각 Hard의 6상자 입력 구조로 확장하고, 같은 Hard 개발 시드에서
첫 집기와 분류 점수를 비교합니다. 더 나은 한 모델만 골라 기존 Hard 성공/복구 시연 24개로 최대
6,000회 저학습률 보정합니다. 실패한 Hard v1/v2 결과와 Release는 그대로 보존합니다.

Hard v3는 별도 실행 `moveboxes_hard_transfer_v3`과 별도 GitHub Release를 사용합니다. 기존 Easy와 Medium
결과를 읽기만 하며 변경하지 않습니다. 전이 이후 보정 학습과 평가는 Hard에만 실행합니다.

| 셀 | 작업 |
|---|---|
| 01~05 | 설정, 연결, 설치, Hard 데이터와 전이 모델 준비 |
| 06 | 기존 복원 상태 확인 |
| 07 | Easy/Medium 시작점 비교 후 Hard 보정 학습 |
| 08 | Hard 8회 테스트, 영상, 행동 기록 |
| 09 | 구간별 성적 확인 |
| 10 | Hard 최종 100회 평가 |
| 11 | Hard 모델 패키지 |

코드는 `hard/code/`, 노트북은 `hard/notebooks/`, 검사는 `hard/tests/`에 있습니다.
