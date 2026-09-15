# 집기 실패 비교: 성공했던 Stage ACT와 현재 run

2026-09-15 확인. 과거 모델은 검사에만 사용했고 현재 run/제출 checkpoint는 수정하지 않았습니다.

## 확인한 파일

| 항목 | 성공 기록이 있는 Easy Stage ACT | 현재 Easy |
|---|---|---|
| Release | `run-moveboxes_easy_lab_v1_benchmark` | `run-moveboxes_stage_chunk_deadline_v1_benchmark` |
| checkpoint | `block_03.pt`, 6,000 step | `latest.pt`, 12,000 step 완료 |
| SHA256 | `0fbbef92aacc5cbe5642568ec5bd349979934e50f3d03d5120ccd5350720ef19` | `a947edb7d37a4e15fa18cc5e38bc379e18367ce74d9b8f9f8e3873a1e454a043` |
| 구조 | state54/history16/chunk16/width128/heads4/layers2/latent16 | 동일 |
| 학습 경로 | zero-latent prior 직접 학습 | 동일 |
| 학습 코드 SHA256 | `2661a2fff8d2190676070e2fed7b37e920ba6c7bb631be35c42159208d71ae4d` | 동일 |
| 초기화 | 기존 StageACT warm start | 처음부터 새 run 학습 |
| 첫 집기 집중 표집 | 30% | 0% |
| warmup | 100 step | 500 step |
| 학습률 / batch | 1e-4 / 64 | 동일 |
| XYZ 실행 | 매 step 재추론, temporal ensemble4 | pick2/carry6/place2/done1 chunk 실행 |
| gripper 실행 | 매 step 최신 예측 | cached 미래 예측 + margin .5 / 2-step 확인 |

성공 run의 마지막 저장 job은 8,000 step 경계이며 선택 checkpoint는 6,000 step입니다. 20,000은 설정된 최대 예산이며 실제 선택 모델 학습량이 아닙니다. 학습량만 늘리면 나아진다는 결론은 나오지 않습니다.

## 같은 관측을 넣은 CPU 재생

현재 recovery 시연 4개를 그대로 순서대로 넣었습니다. 로봇의 다음 상태는 전문가 시연에서 가져오는 검사이므로 실제 시뮬레이터 성공률이 아닙니다. 자동 승자 선택도 없습니다.

`episode_100000.npz`에서 확인한 예:

| 시점 | 전문가 / 참고 모델 | 현재 모델의 기존 실행 | 수정 실행 |
|---|---|---|---|
| 첫 집기, step20 | gripper 닫기 -1 | 열기 +1 | 닫기 -1 |
| 첫 놓기, step51 | gripper 열기 +1 | 닫기 -1 | 열기 +1 |
| 두 번째 집기, step81 | gripper 닫기 -1 | 열기 +1 | 닫기 -1 |
| 두 번째 놓기, step112 | gripper 열기 +1 | 닫기 -1 | 열기 +1 |

동일 step19에서 참고 모델의 native Z 명령은 -0.01361, 현재 native Z는 +0.02908입니다. 두 번째 접촉 직전 step80에서도 참고 모델은 -0.03201, 현재 모델은 +0.00289입니다. 방향이 바뀐 실제 출력 사례이며 gripper 필터만으로 전체 조준 문제가 해결된다는 근거는 아닙니다.

## 수정과 다음 확인

- XYZ horizon은 유지합니다. `fresh_gripper=True`이면 매 step 현재 관측으로 gripper를 예측하고 같은 decode 결과를 XYZ replan에도 재사용합니다.
- 생성 candidate는 `integrated_candidate_v2`입니다. 기존 `integrated_candidate`와 학습 가중치를 보존합니다. 새 기본 정책은 `gripper_fsm=False`여서 접촉/놓기 명령에 2-step 지연을 추가하지 않습니다.
- 별도 `moveboxes_stage_execution_check_colab.ipynb`는 기존 01~05 준비 이후 학습 없이 현재 가중치를 공식 평가하고 영상을 보여줍니다. 참고 Easy 모델도 동일 `default.yaml`로 검사하며 제출본에는 반영하지 않습니다.
- 현재 runtime은 사용자가 중단했으며 수정본의 실제 T4 분류 성능은 아직 확인하지 못했습니다. Easy 조준 문제가 계속되면 추가 학습 전에 XYZ 실행과 접촉 구간 학습을 더 점검해야 합니다.
- 선택 fine-tune 노트북은 기본 `RUN_FINE_TUNE=False`입니다. 현재 부모 run과 참고 모델을 자동 교체하지 않습니다.

## 사용자 요청에 따른 새 가중치 재학습

최종 작업은 실패 모델을 이어 학습하는 방식에서 **처음부터 별도 run을 학습하는 방식**으로 변경했습니다. `moveboxes_stage_all_pick_retrain_colab.ipynb`의 01~05는 기존과 동일하고 부모 run에서는 시연만 가져옵니다. 초기 checkpoint를 가져오는 코드는 없습니다.

새 run은 `_all_pick_retrain_v1`입니다. 모델 구조는 성공했던 StageACT와 같고 batch64/lr1e-4/warmup100/prior 직접 학습을 사용합니다. 각 난이도 12,000 step, 500 step마다 원래 trainer의 optimizer/scaler/RNG 전체 저장과 GitHub 동기화를 수행합니다. 실행도 성공했던 native StagePolicy로 되돌려 매 step XYZ를 재추론하고 temporal ensemble4를 사용하며 최신 gripper를 바로 적용합니다. act_horizon은 학습·실행 모두 1입니다. 이 새 run에는 pick2/carry6 chunk 실행을 적용하지 않습니다.

학습 배치의 기본 50%는 첫 상자와 이후 모든 상자의 PICK 단계입니다. 시연에서 처음 만난 target 순서를 그룹으로 만들고 그룹마다 같은 표집 기회를 부여합니다. 같은 target 재집기는 새 상자로 세지 않습니다. 나머지 배치에는 기존 전체/복구/전환 표집을 유지합니다. 미래 chunk는 기존과 같이 stage/target 경계와 perturbation 뒤에서 마스크하므로 다른 상자 행동이 이어 붙지 않습니다. 검증에는 집중 표집을 적용하지 않습니다.

현재 모델 가중치를 가져오지 않는 준비 경로, 모든 다음 target 포함/재집기 구분, sampler 난수와 optimizer의 정확한 중단 복구, 학습 horizon1/native 실행과 episode 경계 초기화를 CPU에서 검사했습니다. 실제 T4 재학습과 공식 점수 확인은 새 Colab에서 수행해야 합니다.

## Git의 두 번째 집기 데이터

`run-moveboxes_next_pick_v02_benchmark`의 Easy snapshot에는 `sampling_config.json`, `sampling_audit.json`, `next_pick_sampling.py`가 있습니다. 별도 수집 데이터가 아닌 기존 시연의 두 번째 이후 안정적 grasp-cycle 주변 가중 표집입니다. 같은 상자를 다시 잡는 경우도 있어 두 번째 상자 정답 라벨로 간주할 수 없습니다.

기존 StageACT의 Easy recovery 16개는 두 상자를 모두 분류한 성공 시연입니다. 현재 collection manifest와 과거 collection manifest가 동일하며 파일 해시도 같습니다. 두 번째 상자 시연은 현재 학습 데이터에 이미 포함돼 있습니다. 중복 다운로드로 데이터량이 증가했다고 보고하면 안 됩니다.

Medium `curriculum-v21`도 별도로 존재합니다. 저장된 32개 중 전체 4상자 성공은 0개이므로 완전 성공 데이터로 표시하거나 현재 full-success recovery manifest에 무조건 합치지 않습니다.
