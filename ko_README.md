

# AI 생성 텍스트 탐지 프로젝트

이 프로젝트는 주어진 텍스트가 AI에 의해 생성되었는지, 아니면 사람이 작성했는지를 탐지하는 것을 목표로 합니다. 분류를 위해 Ko-ELECTRA 모델을 사용합니다.

## 프로젝트 구조

```
├───data/
│   ├───sample_submission.csv
│   ├───test.csv
│   └───train.csv
├───models/
│   ├───best_Ada_model/
│   ├───best_extra_model/
│   ├───koelectra_42_4000_model/
│   └───koelectra_9000_model/
├───notebooks/
│   └───3rd super_Ada.ipynb
├───processed_data/
│   ├───test_ds/
│   └───train_ds/
├───src/
│   ├───data_preparation.py
│   ├───electra.py
│   ├───ensemble.py
│   ├───inference.py
│   ├───modeling.py
│   ├───superloss.py
│   ├───train.py
│   ├───train_9000.py
│   └───train_super_ada.py
├───.python-version
├───ko_README.md
├───README.md
└───requirements.txt
```

- **`data/`**: 원본 데이터 파일(`train.csv`, `test.csv`, `sample_submission.csv`)이 들어있습니다.
- **`models/`**: 학습된 로컬 모델 체크포인트를 저장합니다. 사전 학습된 모델들은 허깅페이스(Hugging Face)에 업로드되어 있습니다: `218jw/best_Ada_model`, `218jw/best_extra_model`, `218jw/koelectra_42_4000_model`, `218jw/koelectra_9000_model`.
- **`notebooks/`**: 실험에 사용된 주피터 노트북이 들어있습니다.
- **`processed_data/`**: 전처리 및 토큰화된 데이터셋을 저장합니다.
- **`src/`**: 모든 소스 코드가 들어있습니다.
  - `data_preparation.py`: 원본 데이터를 전처리하고 토큰화하는 스크립트입니다.
  - `electra.py`: 토큰 길이 필터링을 포함한 Ko-ELECTRA 모델의 특정 학습 스크립트입니다.
  - `ensemble.py`: 여러 예측 결과(CSV)를 소프트 보팅 방식으로 앙상블하는 스크립트입니다.
  - `inference.py`: 학습된 모델을 사용하여 테스트 데이터에 대한 추론을 실행하는 스크립트입니다.
  - `modeling.py`: `SuperLossElectraForSequenceClassification`과 같은 사용자 정의 모델 정의를 포함합니다.
  - `superloss.py`: SuperLoss 함수의 구현체입니다.
  - `train.py`: 다양한 설정으로 모델을 학습시키기 위한 범용 스크립트입니다.
  - `train_9000.py`: 랜덤 시드를 고정하지 않고 전체 데이터 9000개를 학습하기 위한 변형 스크립트입니다.
  - `train_super_ada.py`: `3rd super_Ada.ipynb` 노트북에서 변환된 특정 학습 스크립트입니다.
- **`.python-version`**: 프로젝트에서 사용하는 파이썬 버전 명시 파일입니다.
- **`ko_README.md`**: 이 프로젝트에 대한 한국어 설명 파일입니다.
- **`README.md`**: 이 프로젝트에 대한 영어 설명 파일입니다.
- **`requirements.txt`**: Python 종속성 목록 파일입니다.

## 설정

1.  **의존성 설치:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **학습 데이터 배치:**

    `train.csv` 파일이 `data/` 디렉토리에 있는지 확인하세요.

## 실행 방법

### 1. 데이터 준비

먼저 원본 데이터를 전처리해야 합니다. 이 스크립트는 `data/train.csv`를 읽고, 훈련 및 평가 세트로 분할한 다음, 텍스트를 토큰화하여 `processed_data/` 디렉토리에 결과를 저장합니다.

```bash
python src/data_preparation.py \
    --input_csv data/train.csv \
    --output_dir processed_data \
    --dataset_name tokenized_ds
```

### 2. 학습

데이터 준비가 완료되면 모델을 학습시킬 수 있습니다. `train.py` 스크립트는 범용 학습에 사용됩니다. `--use_superloss` 플래그를 추가하여 `SuperLoss`를 활성화할 수 있습니다.

**일반 학습:**

```bash
python src/train.py \
    --output_dir models/my_electra_model \
    --processed_data_path processed_data/tokenized_ds
```

**SuperLoss를 사용한 학습:**

```bash
python src/train.py \
    --output_dir models/my_superloss_model \
    --processed_data_path processed_data/tokenized_ds \
    --use_superloss
```

또는 원본 노트북의 특정 학습 구성을 실행할 수 있습니다:

```bash
python src/train_super_ada.py
```

### 3. 추론

학습 후 `inference.py` 스크립트를 사용하여 테스트 세트에 대한 예측을 생성할 수 있습니다. 로컬 모델 경로 또는 허깅페이스 모델 저장소 경로(예: `218jw/koelectra_9000_model`)를 제공해야 합니다.

```bash
python src/inference.py \
    --checkpoint 218jw/koelectra_9000_model \
    --test_csv data/test.csv \
    --output submission
```

이 명령어는 루트 디렉토리에 `submission.csv` 파일을 생성합니다.

### 4. 앙상블 (Ensemble)

4개의 모델(Ada, Extra 등)에 대한 추론 결과(CSV)가 준비되었다면, `ensemble.py` 스크립트를 사용하여 예측 확률의 평균(Soft Voting)을 통해 앙상블을 수행할 수 있습니다.

```bash
python src/ensemble.py \
    --csvs sub_ada.csv sub_extra.csv sub_42_4000.csv sub_9000.csv \
    --output final_ensemble_submission.csv
```

성공적으로 완료되면 `final_ensemble_submission.csv` 파일이 생성됩니다.
