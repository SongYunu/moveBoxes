# 예전 Medium 40.6% 모델을 Easy에 그대로 평가

[Colab 열기](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/unified/notebooks/moveboxes_medium_block02_easy_test.ipynb)

01~05 준비 후 **06 Easy 평가**, 필요하면 **07 Medium 평가**를 실행합니다. 학습 셀은 없습니다.
Medium Lab 4,000회 학습 `block_02.pt` 원본 파일을 그대로 불러오고 SHA-256
`f7c0d5edf5b2bd6b93918d10af43b9c4ee023befd3162139e391b371e1541730`을 검사합니다.
모델 구조와 가중치는 바꾸지 않습니다. Easy의 없는 상자 2개에 해당하는 관측 슬롯만 0으로 채워
54차원 관측을 원래 Medium 모델의 72차원 입력으로 맞춥니다.

과거 40.625%는 시드 40000~40007의 **Medium 분류 점수**입니다. 이번 Easy 평가의 기대 점수가 아닙니다.
이번 노트북의 기본 테스트 시드는 330000~330007입니다. 07 Medium 평가도 시드가 달라 과거 점수와
다를 수 있습니다. Easy의 없는 상자 슬롯은 기존 Medium 훈련 분포와 달라 실패할 수 있습니다.

별도 run_name/코드 폴더/환경 코드 폴더를 사용해 진행 중인 Medium 학습 파일을 덮어쓰지 않습니다.
CPU에서 원본 파일 바이트 보존, 원래 Medium 관측에 대한 행동 출력 일치, Easy 관측 패딩과
노트북 로딩을 검사했습니다. 실제 Easy 물리 성공률은 06에서 확인합니다.

Chrome 창을 나누는 것 자체가 모델이나 가중치를 바꾸지는 않습니다. 다만 다른 Colab 런타임은
메모리·설정·설치·로드된 파일이 별개입니다. 같은 GPU 런타임에서 학습과 평가를 동시에 돌리면
자원을 경쟁하므로, 진행 중인 학습을 마친 뒤 실행하거나 별도 런타임에서 평가하세요.
