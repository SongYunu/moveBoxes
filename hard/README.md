# Hard 전용 실행

[Hard 전용 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/hard/notebooks/moveboxes_hard_colab.ipynb)을
Colab 2026.07 / Python 3.12 / T4에서 실행합니다.

현재 권장 버전은 [Hard v2 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/hard/notebooks/moveboxes_hard_v2_colab.ipynb)입니다.
Hard v1에서 수집한 성공/복구 시연 24개를 재사용하고, 어느 상자를 집을지도 학습해 해당 상자의
위치·회전으로 행동을 조건화합니다.

Hard는 별도 실행 `moveboxes_hard_stage_v1`과 별도 GitHub Release를 사용합니다. 기존 Easy와 Medium
결과를 건드리지 않습니다. 집기 실패 뒤 복구를 포함하는 단계 ACT 시연 수집과 학습을 Hard에만 실행합니다.

| 셀 | 작업 |
|---|---|
| 01~05 | 설정, 코드, GitHub 연결, 설치, 데이터 |
| 06 | 100회 동작 확인 |
| 07 | 실패와 복구가 포함된 성공 시연 수집 |
| 08 | Hard 학습 |
| 09 | 8회 테스트, 영상, 행동 기록 |
| 10 | 튜닝과 최종 평가 |
| 11 | 결과 확인 |
| 12 | 제출용 패키지 |

코드는 `hard/code/`, 노트북은 `hard/notebooks/`, 검사는 `hard/tests/`에 있습니다.
