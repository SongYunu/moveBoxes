# RGB foundation-model experiments

Easy·Medium·Hard의 RGB 시연 600개를 한 모델에 함께 넣는 두 개의 독립 백엔드입니다.
세 노트북은 같은 데이터 분할과 공식 RGB 환경 평가를 사용하지만 모델 환경은 완전히 분리합니다.

**권장 진입점:** [두 모델 통합 Colab](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/foundation/notebooks/moveboxes_foundation_both_colab.ipynb). 이 노트북 하나가
SmolVLA와 Octo를 T4에서 순서대로 학습·평가하고 결과를 비교합니다. 한 백엔드가 설치나 메모리 확인에서
실패해도 다른 백엔드는 계속 실행합니다. 두 모델의 action을 평균내는 ensemble은 아니며, 고정 개발
seed의 `0.2×Easy + 0.3×Medium + 0.5×Hard`가 높은 checkpoint를 후보로 표시합니다.

| 실험 | 공개 초기값 | 학습 형식 | Colab |
|---|---|---|---|
| SmolVLA | `lerobot/smolvla_base` (약 450M) | PyTorch, vision/VLM 동결, action expert adapter `.pt` | [열기](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/foundation/notebooks/moveboxes_smolvla_colab.ipynb) |
| Octo Small 1.5 | `rail-berkeley/octo-small-1.5` (27M policy) | JAX/Flax, 언어·vision encoder 동결, adapter `.npz` | [열기](https://colab.research.google.com/github/SongYunu/moveBoxes/blob/main/foundation/notebooks/moveboxes_octo_colab.ipynb) |

두 모델은 state 전용 HDF5가 아니라 `scene_camera/rgb`가 들어 있는 RGB HDF5를 씁니다. 입력은
128×128 장면 영상, 26차원 proprioception, 고정 작업 문장입니다. 출력은 환경의 4차원
`pd_ee_delta_pos` action입니다. Octo의 원래 7차원 action 중 회전 3축은 loss에서 제외하고
XYZ와 gripper만 연결합니다.

여기서 26차원 proprioception은 `qpos 9 + qvel 9 + tcp_pose 7 + is_grasped 1`입니다.
별도 state ZIP의 54차원 privileged observation은 다운로드하거나 입력하지 않습니다. RGB 평가 환경에서
제공되지 않는 privileged 값을 학습에 넣으면 실제 평가 입력과 달라지기 때문입니다.

Colab에서 **Python 3.12 / T4**를 선택하고 01~06을 먼저 실행합니다. 04번 셀은 현재 커널에
모델 패키지를 섞지 않습니다. SmolVLA는 별도 Python 3.12 환경, Octo는 별도 Python 3.10/JAX
환경, ManiSkill 평가는 별도 Python 3.12 환경에 설치됩니다. 개별 노트북의 01번 `backend`를
임의로 바꾸지 말고, 두 모델을 함께 돌릴 때는 통합 노트북을 사용하세요.

06번 셀은 공개 초기값 다운로드, 세 번의 실제 update, Easy 1회 rollout까지 수행합니다.
메모리 부족이나 API 불일치는 여기서 학습 전에 드러납니다. 07번 셀은 SmolVLA 1,500 update,
Octo 1,900 update마다 고정된
검증 seed로 세 난이도를 모두 평가하고 위 가중 분류율이 높은 adapter만 GitHub Release에 보존합니다.
08~10은 그 **동일한 최고 adapter**를 Easy·Medium·Hard에서 각각 시험합니다.

학습 sampler는 각 에피소드를 action chunk 단위로 빈틈없이 나눈 뒤 세 난이도의 window를 전역으로
섞어 순회합니다. SmolVLA는 16,808개 window를 실효 배치 6으로 3,000 update, Octo는 33,305개를
실효 배치 6으로 5,700 update 학습합니다. 따라서 Easy·Medium·Hard의 모든 train action을 최소
한 번 loss target으로 사용한 checkpoint만 최고 모델 후보가 됩니다. 데이터 양에 따른 자연 비율은
대략 Easy 16%, Medium 32%, Hard 52%라서 Hard를 중시하는 공식 0.2/0.3/0.5 가중치에도 가깝습니다.

실제 시간은 최초 다운로드와 JAX 컴파일, T4 할당 상태에 따라 크게 달라집니다.
06번 셀의 `resource.json`이 해당 런타임의 update 시간과 최고 GPU 메모리를 기록합니다. 보수적으로
SmolVLA는 약 3~7시간, Octo Small은 약 4~10시간을 잡되, 이는 이 저장소에서 아직 T4 실측 완료된
수치가 아닙니다. Colab 종료에 대비해 각 평가 경계의 진행 adapter와 평가 JSON을 Release로
올립니다. 중간 optimizer는 같은 런타임에서만 이어지고, 새 런타임은 저장된 최고 adapter에서 새
optimizer로 계속합니다.
통합판을 전부 실행하면 두 예상 시간의 합과 설치·평가 시간이 필요해 7~18시간가량 걸릴 수 있습니다.

버전은 모델과 코드뿐 아니라 의존성까지 고정합니다. 주요 원본은
[SmolVLA checkpoint](https://huggingface.co/lerobot/smolvla_base),
[LeRobot SmolVLA 문서](https://huggingface.co/docs/lerobot/en/smolvla),
[Octo 코드](https://github.com/octo-models/octo),
[Octo Small checkpoint](https://huggingface.co/rail-berkeley/octo-small-1.5)입니다.
이 구현의 로컬 데이터·shape·재현성 테스트는 실행하지만, 최종 성공률은 Colab T4에서 06~10을
완료한 결과로 판단해야 합니다.
