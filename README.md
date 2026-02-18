# oxford-pet-model-comparison
Oxford-IIIT Pet Dataset을 사용하여 여러 CNN 백본(SimpleCNN, MobileNetV2, ResNet18, EfficientNet-B0)을 동일한 학습 파이프라인에서 비교하기 위한 프로젝트입니다.  
Hydra 기반 설정 관리를 적용하여 코드 수정 없이 다양한 실험을 실행할 수 있도록 구성했습니다.  
데이터 로딩, 모델 생성, 학습 루프, 추론 로직을 분리하여 재현성과 유지보수성을 고려했습니다.

## Installation
editable 모드로 설치하면 코드 수정 시 재설치 없이 바로 반영됩니다.
```bash
pip install -e .
```

## Data Setup
```bash
kaggle datasets download julinmaloof/the-oxfordiiit-pet-dataset
Expand-Archive the-oxfordiiit-pet-dataset.zip -DestinationPath datasets/oxford-iiit-pet
```
경로는 configs/dataset/oxford_pet.yaml에서 변경할 수 있습니다.

## Train (scripts/train.py)  
Hydra CLI override를 통해 다양한 모델 및 실험 설정으로 학습을 실행할 수 있습니다.
```bash
python scripts/train.py -m model=resnet18,mobilenet_v2,efficientnet_b0,simple_cnn
python scripts/train.py model=resnet18 exp=freeze
```

## Output
실험 결과는 out/ 폴더 아래에 저장됩니다.
```text
out/
 └── <exp_name>/
      ├── best.pt
      ├── history.pt
      └── config.yaml
```
best.pt : validation accuracy 기준 최고 성능 모델
history.pt : epoch별 학습 기록
config.yaml : 실행 시 사용된 Hydra 설정

## Evaluate (scripts/test.py)  
```bash
python scripts/evaluate.py model=resnet18 exp=freeze                                      
```

## Predict (scripts/predict.py) 
로컬 이미지
```bash
python scripts/predict.py image_path="sample.jpg" model=resnet18 exp=freeze
```
URL 이미지
```bash
python scripts/predict.py image_path="https://cdn.britannica.com/16/234216-050-C66F8665/beagle-hound-dog.jpg" model=resnet18 exp=freeze
```

## Hydra 기반 설정 관리
실험 설정은 Hydra를 사용하여 YAML 파일로 분리 관리합니다.
- dataset / model / train / exp 설정 분리
- CLI override로 실험 조건 변경 가능
- 실행 시 최종 config 자동 저장

## Engineering Design
여러 모델을 동일한 조건에서 비교할 수 있도록 공통 학습 파이프라인을 중심으로 구조를 설계했습니다.
- build_model()을 통해 모델 생성 관리
- training components는 factory 패턴으로 구성
- train / evaluate loop 분리
- Predictor 클래스로 inference 로직 독립

## Models
현재 구현된 모델:
- SimpleCNN
- MobileNetV2
- EfficientNetB0
- ResNet18
새 모델 추가 시 models/registry.py에만 등록하면 됩니다.

## Project Structure  
```text
oxford-pet-model-comparison/
├─ configs/                      # Hydra 설정들
│  ├─ config.yaml                # 엔트리(기본) config
│  ├─ paths/
│  │  └─ paths.yaml              # data/out 경로 모음
│  ├─ dataset/
│  │  └─ oxford_pet.yaml         # 데이터셋 설정 (num_classes 등)
│  ├─ train/
│  │  └─ train.yaml              # 학습 하이퍼파라미터
│  ├─ exp/
│  │  ├─ base.yaml               # 기본 실험
│  │  └─ freeze.yaml             # freeze_backbone 실험 등
│  └─ model/
│     ├─ simple_cnn.yaml
│     ├─ mobilenet_v2.yaml
│     ├─ resnet18.yaml
│     └─ efficientnet_b0.yaml
│
├─ scripts/                      # 실행 스크립트(진입점)
│  ├─ train.py
│  ├─ evaluate.py
│  └─ predict.py
│
├─ src/oxford_pet_model_comparison/
│  ├─ cli/                       # CLI 실행 로직 (run_train/run_eval/run_predict)
│  │  ├─ __init__.py
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  └─ predict.py
│  │
│  ├─ datasets/                  # dataset / datamodule / transforms
│  │  ├─ __init__.py
│  │  ├─ oxford_pet_dataset.py
│  │  ├─ datamodule.py
│  │  └─ transforms.py
│  │
│  ├─ models/                    # 모델 구현 + registry/build
│  │  ├─ __init__.py
│  │  ├─ registry.py
│  │  ├─ simple_cnn.py
│  │  ├─ mobilenet_v2.py
│  │  ├─ resnet18.py
│  │  └─ efficientnet_b0.py
│  │
│  ├─ training/                  # 학습 루프/트레이너
│  │  ├─ __init__.py
│  │  ├─ trainer.py
│  │  └─ loops/
│  │     ├─ __init__.py
│  │     ├─ train_one_epoch.py
│  │     └─ evaluate_one_epoch.py
│  │
│  ├─ inference/                 # 추론
│  │  ├─ __init__.py
│  │  └─ predictor.py
│  │
│  └─ utils/                     # seed, I/O, 이미지 로딩 등
│     ├─ __init__.py
│     ├─ seed.py
│     ├─ io.py
│     └─ image_utils.py
│
├─ data/                         # (로컬) 데이터 위치
├─ out/                          # 실험 결과 저장 폴더
├─ outputs/                      # hydra 기본 출력 폴더
├─ pyproject.toml
└─ README.md
```
# Architecture Overview
```text
configs/ (Hydra)
       ↓
scripts/train.py
       ↓
datasets/ → models/ → training/
       ↓
out/<exp_name>/
```

# Training Flow
```text
Hydra Config
     ↓
build_datamodule()
     ↓
build_model()
     ↓
Trainer.fit()
     ↓
train_one_epoch / evaluate_one_epoch
     ↓
Checkpoint Save
```
# Inference Flow
```text
predict.py
     ↓
Load best.pt
     ↓
build_model()
     ↓
Predictor.predict()
     ↓
Softmax → class_name / probability
```