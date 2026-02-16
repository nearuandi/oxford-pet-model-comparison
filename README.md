# oxford-pet-model-comparison
oxford-pet-model-comparison

pip install -e .  

python scripts/train.py model=resnet18  
python scripts/train.py exp=freeze model=resnet18  
python scripts/train.py -m model=resnet18,mobilenet_v2,efficientnet_b0,simple_cnn  

kaggle datasets download julinmaloof/the-oxfordiiit-pet-dataset  
Expand-Archive .\the-oxfordiiit-pet-dataset.zip -DestinationPath data\oxford-iiit-pet
