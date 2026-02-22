# oxford-pet-model-comparison

Oxford-IIIT Pet Dataset을 사용한 이미지 분류 실험 레포입니다.
Hydra 기반 설정을 사용해 여러 CNN 모델(SimpleCNN, MobileNetV2, ResNet18, EfficientNet-B0)을 동일한 파이프라인에서 학습·평가·추론할 수 있도록 구성했습니다.

이 프로젝트는 **재현 가능한 실험 구조**와 **CLI 기반 inference 흐름**을 정리하는 것을 목표로 합니다.

---

## Install

```bash
pip install -e .
```

---

## Dataset

```bash
kaggle datasets download julinmaloof/the-oxfordiiit-pet-dataset
Expand-Archive the-oxfordiiit-pet-dataset.zip -DestinationPath datasets/oxford-iiit-pet
```

---

## Train

단일 모델 학습:

```bash
python scripts/train.py model=resnet18
```

여러 모델 멀티 실행:

```bash
python scripts/train.py -m model=resnet18,mobilenet_v2,efficientnet_b0
```

---

## Evaluate

```bash
python scripts/evaluate.py exp=base model=resnet18
```

validation 기준으로 저장된 `best.pt`를 사용해 평가합니다.

---

## Predict

로컬 이미지:

```bash
python scripts/predict.py image_path="cat.jpg"
```

URL 이미지:

```bash
python scripts/predict.py image_path="https://..."
```

출력 예시:

```
cfg.model.name=resnet18
cfg.exp.name=resnet18
class_id=0
class_name=Abyssinian
prob=99.72%
```

---

## Project Structure

```
configs/
scripts/
src/oxford_pet_model_comparison/
  ├─ cli/
  ├─ data/
  ├─ models/
  ├─ engine/
  ├─ pipelines/
  └─ utils/
out/
```

---

## Output

각 실험 결과는 다음 위치에 저장됩니다.

```
out/{exp.name}/
 ├─ best.pt
 ├─ history.pt
 └─ config.yaml
```

* **best.pt** : validation accuracy 기준 최고 성능 모델
* **history.pt** : 학습 기록
* **config.yaml** : 실행 당시 Hydra 설정 (재현성용)

---

## Models

현재 지원 모델:

* simple_cnn
* mobilenet_v2
* resnet18
* efficientnet_b0

새 모델 추가 시:

```
src/oxford_pet_model_comparison/models/build.py
```

에서 등록하면 됩니다.

---

## Hydra Config

기본 구조:

```yaml
defaults:
  - paths: paths
  - dataset: oxford_pet
  - train: train
  - exp: base
  - model: resnet18
```

CLI override 예시:

```bash
python scripts/train.py model=efficientnet_b0 exp=freeze
```

---

## Notes

* Trainer 기반 공통 학습 루프 사용
* AMP mixed precision 지원
* config 자동 저장으로 실험 재현 가능
* 동일 구조에서 모델 비교 목적

---

## License

MIT License
