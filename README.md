# ESS 배터리 수명 예측
최초 5사이클의 충방전 특성을 이용해 단수명 셀을 조기에 분류하고, 배치가 달라져도 성능이 유지되는지 확인한다.

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20)
- 태스크 : Classification (장단수명 분류)
- Target : `cycle_life < 550 → 단수명(0)`, `cycle_life ≥ 550 → 장수명(1)`
- Batch 3 : EDA에만 사용한다.

## 파일 구조
```text
├── data/rawdata/archive/       # 원본 MAT 데이터 (Git 제외)
├── code/
│   ├── EDA.ipynb              # 데이터 탐색
│   ├── model.ipynb            # 피처·모델 비교, 평가 및 오류 분석
│   ├── pipeline.py            # 피처 추출, 전처리, 학습·평가 함수
│   └── run_pipeline.py        # 노트북 전체 재실행
├── info/                      # 설계 보고서
├── results/
│   └── model_performance.csv  # 후보별 성능과 평가기준의 Gap을 통합
├── requirements.txt
└── README.md
```

## 환경 설정
Python 3.11 기준이다. 저장소를 받은 뒤 프로젝트 루트에서 실행한다.
```bash
git clone https://github.com/yyun361/DS-Miniproject.git
cd DS-Miniproject
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
노트북을 직접 실행할 경우 `.venv`의 Python 커널을 선택하고 `code/model.ipynb`를 위에서부터 실행한다.
재실행하면 노트북 출력과 `results/model_performance.csv`를 갱신한다.
실행에 필요한 커널·그래프 설정 파일은 운영체제 임시 폴더에 두고 실행 종료 후 제거한다. 프로젝트에 `work/`는 만들지 않는다.
피처, 예측, 오류 목록과 혼동행렬은 노트북에서 확인하며 별도 파일로 저장하지 않는다.

## 데이터 준비와 저장된 결과
원본 MAT 데이터는 `.gitignore`의 `data/` 규칙으로 Git에서 제외한다. 저장소를 clone한 뒤 데이터를 별도로 준비한다.

1. 이 프로젝트에서 사용한 [Kaggle MIT-Stanford Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle) 페이지에서 데이터를 다운로드한다. 로그인 요청이 표시되면 Kaggle 계정으로 로그인한다.
   원본 연구 데이터의 출처는 [MATR](https://data.matr.io/1/)이며, [원논문 공식 코드 저장소](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)에서도 출처를 확인할 수 있다.
2. 압축 파일로 받았다면 압축을 풀고, 아래 폴더에 `.mat` 파일을 직접 넣는다. 경로는 저장소 루트 기준이다.

```text
DS-Miniproject/
└── data/
    └── rawdata/
        └── archive/
            ├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
            ├── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
            └── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

- Batch 1: 학습 및 내부 검증에 사용한다.
- Batch 2: 최종 평가에 사용한다.
- Batch 3: EDA에만 사용한다.

위 파일명은 실제로 사용한 Kaggle 다운로드 파일을 기준으로 한다. 압축을 풀었을 때 폴더가 중첩되어 있으면 `.mat` 파일을 위 위치로 옮긴다. 추가 파일은 그대로 두어도 되며 코드는 위 세 파일을 이름으로 지정해 읽는다.

3. 데이터 준비 후 프로젝트 루트에서 모델 노트북을 실행한다.

```bash
python code/run_pipeline.py
```

EDA는 `code/EDA.ipynb`를 위에서부터 실행한다. 모델 실행에는 Batch 1·2가 필요하고, 전체 EDA 실행에는 세 Batch가 모두 필요하다.

`results/model_performance.csv`와 실행 결과를 포함한 노트북은 Git에 함께 올린다. 데이터 없이도 저장된 결과를 열람할 수 있지만, 데이터 로딩부터 재실행하려면 원본 파일이 필요하다.
재실행은 결과 CSV를 덮어쓴다. `results` 폴더 생성 코드는 새 checkout에서도 결과를 저장하기 위한 코드다.
Windows에서는 가상환경 활성화 명령으로 `.venv\Scripts\Activate.ps1`을 사용한다.

## Data Leakage 점검 범위
학습 함수는 Batch 1만, 최종 평가 함수는 Batch 2만 받도록 검사한다. 모델 선택에는 Batch 2 결과를 전달하지 않는다.
결측 대체와 표준화는 sklearn Pipeline 안에서 학습 데이터에만 fit한다. 입력은 초기 1~5사이클 및 충전 정책으로 제한한다.
다만 Batch 접두사가 있는 `cell_id`의 분리는 실제 물리적 배터리의 중복 여부를 증명하지 않는다.
원본 데이터의 Batch 간 연속 시험 셀 매핑은 별도로 확인해야 하며, 현재 코드는 원본 파일의 셀들을 별개로 취급한다.
Batch 2의 모든 후보 성능은 사후 비교용이며, 이 결과로 모델·피처·임계값을 재선택하면 독립 테스트로 사용할 수 없다.

## EDA
- Cycle Life 분포
  - Batch 1 : 단수명 1개·장수명 45개, Batch 2 : 단수명 30개·장수명 9개
  - Batch 2의 수명 결측 8개는 수명 비교·분류에서 제외한다.
  - 핵심 발견 : 학습과 테스트의 클래스 구성이 달라, 높은 학습 성능이 일반화를 보장하지 않는다.
- 열화 곡선 분석
  - Batch 2는 용량 감소가 빠르고 knee가 더 일찍 나타나는 경향이 있다.
  - 핵심 발견 : 열화 속도는 수명 신호가 될 수 있지만 전체 기울기·knee는 미래 정보라 입력에서 제외한다.
- ΔQ(V) 곡선 분석
  - Cycle 100-10 차이에서 단수명 셀의 변화가 큰 경향을 확인했다.
  - 핵심 발견 : 이 신호를 그대로 사용하지 않고 초기 5사이클 내 `Q5-Q4`로 다시 계산한다.
- 충전 속도(C-rate)와 수명의 관계
  - 첫 단계 C-rate와 수명의 관계는 배치별로 다르며, 같은 C-rate에서도 정책 조합에 따라 수명이 달랐다.
  - 핵심 발견 : 첫 C-rate·전환 SOC·둘째 C-rate를 함께 비교하되 인과관계로 단정하지 않는다.
- 피처 간 상관관계
  - ΔQ 로그분산과 최솟값은 중복 정보일 가능성이 있다.
  - 핵심 발견 : 두 피처를 함께 사용한 조합과 로그분산만 사용한 조합을 비교한다.

## Modeling
### Feature Engineering 전략
초기 5사이클의 상태와 변화 정도를 요약하고, EDA에서 확인한 ΔQ 및 충전 정책 정보의 추가 효과를 비교한다.
- 기본 6개 : 1~5사이클 방전용량 평균·표준편차, IR·Tavg·Tmax·충전시간 평균
- ΔQ 2개 : `Q5(V)-Q4(V)`의 로그분산과 최솟값
- 정책 3개 : 첫 C-rate, 전환 SOC(%), 둘째 C-rate
- 비교 조합 : 기본 / 기본+ΔQ / 기본+로그분산 / 기본+ΔQ+정책

빈 측정 행은 제거하고, 용량 0~1.43Ah 밖과 IR·충전시간 0 이하는 결측 처리한다.
빈 첫 사이클을 6사이클로 대체하지 않는다. 보완(학습 중앙값)·표준화·분류기는 하나의 Pipeline으로 묶는다.
수명, Batch, cell_id, 전체 열화 기울기, knee와 100사이클 피처는 입력에서 제외한다.

### 모델 선택 및 근거
- 후보 모델 : 정규화 Logistic Regression, RBF SVM, Random Forest, Shrinkage LDA. Dummy는 비교 기준이다.
- 최종 모델 : **Logistic + 기본·ΔQ 8개 피처를 잠정 기준 모델로 유지한다.** 검증된 최적 모델은 확정하지 못했다.
- 선택 이유 : 소수 표본에서 복잡도를 제한하고, 클래스 가중치와 규제를 적용하는 해석 가능한 기준 모델이다.

설계 후보에 제한하지 않고 Shrinkage LDA를 추가했다. Shrinkage LDA는 소표본과 피처 간 중복에 대응하기 위한 추가 후보 모델로 사용한다. 다만 단수명 표본이 1개뿐이므로 클래스 특성을 안정적으로 추정하기 어렵다는 한계가 있다.

같은 충전 정책이 학습·검증에 중복되지 않도록 Batch 1을 그룹 Hold-out으로 나눈다.
시드 42에서 Fit 35개(단수명 1), Valid 11개(장수명만), Test 39개다.
Batch 1 단수명 셀이 하나뿐이라 두 클래스를 독립적으로 포함하는 CV와 Hold-out 검증은 불가능하다.
후보·피처 조합 대부분이 Valid에서 100%지만 Dummy도 같아 후보의 우열이나 피처 개선 효과를 확정할 수 없다.
Batch 2 결과로 모델을 다시 선택하거나 설정을 조정하지 않는다.

## 성능 결과
F1은 장수명(1) 기준이다. 아래 Accuracy·Recall은 %이며, CSV에는 0~1 비율로 저장한다.

| 후보 모델 | Test F1 | Test Accuracy | 단수명 Recall | Balanced Accuracy |
|---|---:|---:|---:|---:|
| Dummy | 0.3750 | 23.08% | 0.00% | 50.00% |
| Logistic (잠정 기준) | 0.2667 | 15.38% | 0.00% | 33.33% |
| RBF SVM | 0.3750 | 23.08% | 0.00% | 50.00% |
| Shrinkage LDA | 0.2222 | 28.21% | 23.33% | 33.89% |
| Random Forest | 0.3043 | 17.95% | 0.00% | 38.89% |

Shrinkage LDA가 후보 중 가장 높은 Accuracy와 단수명 Recall을 보였지만 Balanced Accuracy는 50% 미만으로, 안정적인 개선으로 보기는 어렵다. 따라서 Test 결과만으로 최종 모델을 변경하지 않는다.

잠정 Logistic 모델의 평가표는 다음과 같다. CSV에는 모든 후보의 같은 평가표를 통합했다.

| 구분 | F1-Score | Accuracy | 비고 |
|---|---:|---:|---|
| Train (Batch 1 CV) | N/A | N/A | 단수명 1개로 계산 불가 |
| Valid (Batch 1 Hold-out) | 1.0000 | 100.00% | 장수명 11개만 포함 |
| Test (Batch 2) | 0.2667 | 15.38% | 최종 평가 39개 |
| Gap (Train-Valid) | N/A | N/A | CV 불가 |
| Gap (Valid-Test) | 0.7333 | 84.62%p | 클래스 구성 차이 포함 |
| Gap (Target-Test) | N/A | 79.72%p | Target Accuracy 95.1% |

Gap은 앞 평가 구간에서 뒤 평가 구간의 성능을 뺀 값이다. Batch 1의 단수명 표본 부족으로 CV는 계산하지 못했으며, Valid 100%도 장수명만 포함된 결과이므로 일반화 성능을 의미하지 않는다. 원논문의 95.1% Accuracy는 조건 차이로 인해 참고 기준으로만 사용한다.

## 오류 분석
- 잠정 Logistic 모델은 단수명 30개를 모두 장수명으로 오판했고 장수명 9개 중 3개를 단수명으로 오판했다.
- 단수명 오류가 여러 충전 정책에서 발생해 특정 C-rate가 원인이라고 단정하기 어렵다.
- 원인 가설 : 단수명 학습 사례 부족과 배치별 초기 측정 특성 차이로 단수명 경계를 충분히 학습하지 못했다.
- 개선 방향 : 단수명 학습·검증 셀을 먼저 확보하고, 독립 배치에서 측정 조건을 확인한 뒤 피처와 모델을 검증한다.

## ESS 도메인 해석
- 활용 의사결정 : 향후 성능이 충분히 확보된다면, 초기 시험에서 단수명 가능 셀을 추가 검사 대상으로 선별하는 데 활용할 수 있다.
- 현재 한계 : 단수명을 장수명으로 오판하면 선별에서 놓친다. 현재 성능은 이러한 실제 운영 판단에 활용하기 부족하다.
- 실 배포 전 필요 사항 : 대표성 있는 셀 확보, 두 클래스의 독립 검증, 배치별 측정 조건 정렬, 오류 비용 기반 임계값 검증이 필요하다.

## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391. https://doi.org/10.1038/s41560-019-0356-8

