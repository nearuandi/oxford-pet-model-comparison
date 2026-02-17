# oxford-pet-model-comparison
Oxford-IIIT Pet Dataset을 기반으로 Hydra 설정을 통해 다양한 실험(exp)을 구성하고 여러 CNN 모델(SimpleCNN, MobileNetV2, ResNet18, EfficientNet-B0)의 학습 파이프라인을 관리하기 위한 프로젝트입니다.  
학습 파이프라인은 재현성과 유지보수성을 고려하여 datasets / engine / models 구조로 분리하였으며, Hydra 기반 설정 관리 시스템을 적용했습니다.

## Installation
현재 프로젝트를 editable 모드로 설치하여 코드 수정 시 재설치 없이 바로 반영됩니다.
```bash
pip install -e .
```

## Data Setup
```bash
kaggle datasets download julinmaloof/the-oxfordiiit-pet-dataset
Expand-Archive the-oxfordiiit-pet-dataset.zip -DestinationPath data/oxford-iiit-pet
```

## Train (scripts/train.py)  
모델 학습을 실행합니다.  
Hydra CLI override를 통해 다양한 모델 및 실험 설정으로 학습을 수행할 수 있습니다.
```bash
python scripts/train.py -m model=resnet18,mobilenet_v2,efficientnet_b0,simple_cnn
python scripts/train.py -m model=resnet18 exp=freeze
```

## Output
각 실험 결과는 runs/ 폴더 아래에 저장되며, validation accuracy 기준으로 가장 성능이 높은 모델과 학습 기록, 그리고 실험 재현을 위한 설정이 함께 보관됩니다.
```text
runs/
 └── <exp_name>/
      ├── best.pt
      ├── history.pt
      └── config.yaml
```

## Test (scripts/test.py)  
저장된 best 모델을 로드한 뒤 test_loader를 사용하여 최종 테스트 성능을 평가합니다.

## Visualize (scripts/visualize.py)  
학습 과정에서 저장된 history를 기반으로 다음 그래프를 출력합니다. 
- Train / Validation Loss
- Train / Validation Accuracy
- Best Validation Accuracy

여러 모델(SimpleCNN, MobileNetV2, ResNet18, EfficientNetB0)의 Validation 성능 곡선 비교도 가능합니다.  
특정 실험(exp)의 결과를 시각화하려면 Hydra CLI override를 사용합니다.
```bash
python scripts/visualize.py model=resnet18 exp=freeze 
```

## Hydra 기반 설정 관리
실험 설정은 Hydra를 사용하여 YAML 파일로 분리 관리합니다.

- dataset / model / train / exp 설정을 개별 파일로 구성
- CLI override를 통해 다양한 실험 설정을 유연하게 변경 가능
- 코드 수정 없이 YAML 및 CLI만으로 실험 조건 제어
- 실행 시 사용된 설정(config.yaml)은 runs 폴더에 자동 저장되어 실험 재현성을 보장합니다.

## Engineering Design
여러 모델을 동일한 조건에서 비교하기 위해 공통 학습 루프를 중심으로 구조를 설계했습니다.  
- 모델 생성은 build_model()에서 관리
- training components는 factory에서 생성
- train/evaluate loop 분리

datasets / engine / models 역할을 나누어 코드 가독성과 유지보수성을 고려했습니다.  
Hydra를 사용하여 실험 설정을 코드에서 분리하고, 재현 가능한 실험 구조를 구성했습니다.

## Models
현재 구현된 모델:
- SimpleCNN
- MobileNetV2
- ResNet18
- EfficientNetB0

새로운 모델을 추가할 경우, model_factory에만 등록하면 학습 코드 수정 없이 확장 가능합니다.

## Project Structure  
```text
oxford-pet-model-comparison
├─ configs
│  ├─ dataset
│  │  └─ oxford_pet.yaml
│  ├─ exp
│  │  ├─ base.yaml
│  │  └─ freeze.yaml
│  ├─ model
│  │  ├─ efficientnet_b0.yaml
│  │  ├─ mobilenet_v2.yaml
│  │  ├─ resnet18.yaml
│  │  └─ simple_cnn.yaml
│  ├─ paths
│  │  └─ paths.yaml
│  ├─ train
│  │  └─ train.yaml
│  └─ config.yaml
│
├─ data
├─ multirun
├─ outputs
├─ runs
│
├─ scripts
│  ├─ train.py
│  ├─ test.py
│  ├─ predict.py
│  └─ visualize.py
│
├─ src
│  └─ oxford_pet_model_comparison
│     ├─ datasets
│     │  ├─ __init__.py
│     │  ├─ dataloaders.py
│     │  ├─ oxford_pet_dataset.py
│     │  └─ transforms.py
│     │
│     ├─ engine
│     │  ├─ __init__.py
│     │  ├─ trainer.py
│     │  ├─ checkpoint
│     │  │  ├─ __init__.py
│     │  │  ├─ save.py
│     │  │  └─ load.py
│     │  ├─ factories
│     │  │  ├─ __init__.py
│     │  │  └─ training_factory.py
│     │  └─ loops
│     │     ├─ __init__.py
│     │     ├─ train_one_epoch.py
│     │     └─ evaluate_one_epoch.py
│     │
│     ├─ models
│     │  ├─ __init__.py
│     │  ├─ model_factory.py
│     │  ├─ resnet18.py
│     │  ├─ mobilenet_v2.py
│     │  ├─ efficientnet_b0.py
│     │  └─ simple_cnn.py
│     │
│     ├─ utils
│     │  ├─ __init__.py
│     │  └─ image_utils.py
│     │
│     └─ visualize
│        ├─ __init__.py
│        └─ plots.py
│
├─ pyproject.toml
├─ LICENSE
├─ README.md
└─ .gitignore
```