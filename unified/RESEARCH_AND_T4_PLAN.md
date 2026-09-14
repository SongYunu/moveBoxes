# 단일 정책·실패 구간 보정·커리큘럼의 근거와 T4 적용 범위

조사일: 2026-09-14. 저자 프로젝트 페이지, 논문 원문/초록, 공식 구현과 GPU 제조사 자료를 확인했다.
MARSO Easy→Medium→Hard와 Colab T4라는 조건까지 동일한 성공 사례는 이번 조사에서 확인하지 못했다.
아래 연구는 구성 요소의 근거이며, 이 조합의 실제 분류 성능을 보장하지 않는다.

## 가장 가까운 사례

### IWR — Human-in-the-Loop Imitation Learning using Remote Teleoperation

- 저자: Ajay Mandlekar 외. [논문](https://arxiv.org/abs/2012.06733), [저자 프로젝트](https://sites.google.com/stanford.edu/iwr).
- 실제 로봇의 threading/coffee manipulation에서 정밀 동작이 필요한 병목을 사람이 교정한다.
- intervention과 non-intervention 표본을 같은 비율로 학습하고, 수집과 학습을 반복한다.
- 우리의 ‘막히는 구간을 집중 보정’과 가장 가깝다. 단, 논문은 **실패 주변의 실제 상태에 대한 교정 행동**을 확보한다.
  기존 정상 시연만 잘라 반복하는 것은 이 연구의 교정 데이터 수집과 다르다.
- T4 적용 판단: 표본 선택·가중치 방식 자체는 큰 별도 모델이 필요하지 않는다. 논문의 로봇 성능과 계산 시간을
  우리 작은 state ACT에 그대로 적용할 수는 없다.

### DAgger — A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning

- Ross, Gordon, Bagnell, AISTATS 2011. [공식 논문 페이지](https://proceedings.mlr.press/v15/ross11a.html).
- 현재 정책이 방문하는 상태에 전문가의 정답 행동을 붙이고 데이터를 누적해 재학습하는 근거다.
- 로봇이 놓쳐 만들어낸 상태가 원래 시연에 없다면, 원래 시연만 더 많이 학습해도 빈틈이 남을 수 있다.
- 실패한 정책이 출력한 행동을 정답처럼 넣는 것은 DAgger가 아니다. 시뮬레이터 교정자도 실패할 수 있으므로
  교정 행동의 유효성을 확인해야 한다. 현재 버전의 전문가 시연 크롭을 DAgger 구현이라고 부르지 않는다.

### ACED — Automatic Curricula via Expert Demonstrations

- Dai, Hofmann, Williams. [논문](https://arxiv.org/abs/2106.09159).
- 전문가 시연을 구간으로 나누고, 종료 지점 가까이에서 시작해 점차 시연 앞쪽으로 초기화 위치를 옮긴다.
- pick-and-place와 block stacking에서 demonstration 기반 커리큘럼과 BC 결합을 실험했다.
- ‘짧은 부분 문제를 해결한 뒤 전체 문제로 확장’의 직접적인 조작 과제 사례다.
- 다만 RL과 시뮬레이터 상태 초기화를 사용한다. 우리의 BC 시간 구간 표집이나 Easy→Medium→Hard
  난이도 순서가 동일 알고리즘이거나 입증된 최적 순서라고 주장할 수 없다. 전체 RL 절차의 T4 비용도 미검증이다.

### LIBERO — Benchmarking Knowledge Transfer for Lifelong Robot Learning

- Liu 외, 2023. [논문](https://arxiv.org/abs/2306.03310), [공식 코드](https://github.com/Lifelong-Robot-Learning/LIBERO).
- 로봇 조작의 순차 학습, 다중 과제 학습, 과제 순서, 지식 전이와 망각을 비교한다. Experience Replay도 비교한다.
- 이전 난이도 데이터를 재사용하고 이전 난이도의 실제 성능도 검사해야 하는 근거다.
- ‘순차 학습은 항상 이긴다’는 근거는 아니다. 논문은 순차 fine-tuning, 사전학습, 알고리즘별 결과가 다름을 보고한다.
- 공식 구현은 GPU 자원이 제한될 때 학습과 평가를 별도로 실행할 수 있게 제공한다. 우리도 이 실행 방식을 따른다.

### ACT — Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

- Zhao 외, RSS 2023. [프로젝트](https://tonyzhaozh.github.io/aloha/), [저자 제공 논문 PDF](https://tonyzhaozh.github.io/aloha/aloha.pdf).
- 행동을 묶음으로 예측하는 imitation learning 모델로 정밀 조작을 다룬다.
- 논문은 RTX 2080 Ti 한 장에서 약 5시간 학습한 설정을 보고한다. **이 수치는 T4 학습 시간 추정치가 아니다.**
- 이 저장소의 state StageACT는 영상 backbone과 원본 ALOHA 구성 전체를 사용하지 않는 별도 축소 변형이다.

## T4를 기준으로 고정할 설계

[NVIDIA 공식 자료](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)에
T4는 16 GB GDDR6와 FP16 혼합 정밀도를 지원하는 것으로 명시되어 있다.

현재 코드로 직접 센 canonical StageACT는 1,528,875 파라미터이다.
설정: state 90, width 128, transformer encoder/decoder 각 2층, heads 4,
history 16, action chunk 16, latent 16. 학습은 현재 실행에서 사용하는 prior 경로를 유지한다.
FP32 파라미터·gradient·Adam 2개 상태만 대략 23.3 MiB이며, 이는 activation·CUDA·시뮬레이터·데이터 메모리를
포함하지 않은 산술 추정이다. 실제 T4 peak VRAM이나 처리량으로 제시하지 않는다.

실행 제약:

- state 입력만 사용하는 동일 모델 하나. Easy/Medium/Hard는 없는 상자 슬롯을 패딩해 공유한다.
- FP16 AMP, microbatch 32에서 시작. 측정 없이 배치를 크게 늘리지 않는다.
- 데이터는 CPU에 두고 현재 minibatch만 GPU로 보낸다. 모든 난이도를 GPU에 통째로 올리지 않는다.
- 학습과 시뮬레이션 평가는 별도 프로세스로 순차 실행한다. 평가는 환경 1개부터 시작한다.
- 처음 100 update로 peak allocated/reserved VRAM과 updates/s를 측정한다.
  이후 시간 추정은 학습 처리량과 episode 평균 평가 시간을 별도로 계산한다.
- 500 update 단위로 제한하고, 개선이 없으면 반복 예산을 무작정 늘리지 않는다.
- 개발 평가는 고정한 소수 시드, 최종 평가는 별도 시드. 매 학습 구간마다 100회 평가하지 않는다.

## 사용자 제안을 적용할 때 지켜야 할 차이

1. Medium checkpoint는 초기 가중치 후보이며 성능이 충분히 검증된 완성 모델이 아니다.
2. Easy 실패를 진단하고 시연의 앞뒤 문맥을 포함한 시간 구간을 보강한다. 이미지 크롭과 다른 의미다.
3. 실제 실패 상태에 맞는 전문가 교정 데이터가 없다면 ‘기존 시연 재가중’으로 명확히 표시한다.
   이 경우 회복 정책에 대한 DAgger/IWR의 근거를 그대로 가져오지 않는다.
4. 통과 여부는 짧은 조각의 loss가 아니라 기본 초기화에서 시작한 전체 에피소드의 정답 분류로 판단한다.
5. Medium에서는 이전 Easy 데이터를 섞고 Easy 성능도 재검사한다. 속도 보너스는 명시적으로 완주가 확인된
   시연끼리만 비교하며, 성공 메타데이터가 없는 시연에는 보너스를 주지 않는다.
6. Hard의 배치·방향 변화는 관측과 데이터가 실제로 포함해야 한다. 단순 차원 확장만으로 적응한다고 가정하지 않는다.

## 현재 구현 상태

`unified/code/`의 입력 패딩·시연 표집·단계별 학습·전체 에피소드 통과·원격 저장을
[Colab 실행 노트북](notebooks/moveboxes_unified_colab.ipynb)에 연결했다.
CPU의 작은 학습과 로딩/분할/진단/단계 차단/패키징 검사를 통과했다.
실제 T4 물리 시뮬레이션의 성공률과 처리량은 미검증이다. 06 셀에서 시간과 VRAM을 측정하고
07~08에서 Easy 성능을 먼저 확인한다. 새 교정 데이터 수집은 포함하지 않으며 성능 향상을 미리 주장하지 않는다.
