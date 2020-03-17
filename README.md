# kaggle-bengali
This respository contains my code for competition in kaggle.

575th Place Solution😇 for [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)

Public score: 0.9804 (188th)  
Private score: 0.9242 (575th)

## Solution
### Overview
- 5-folds Resnet50 (128x128)  

| Fold0 | Fold1 | Fold2 | Fold3 | Fold4 | CV |  
| ----- | ----- | ----- | ----- | ----- | --- |   
| 0.9846 | 0.9860 | 0.9853 | 0.9870 | 0.9873 | 0.9873 |  

- 5-folds efficientnet-b2 (128x128)

| Fold0 | Fold1 | Fold2 | Fold3 | Fold4 | CV |  
| ----- | ----- | ----- | ----- | ----- | --- |   
| 0.9842 | 0.9866 | 0.9839 | 0.9862 | 0.9863 | 0.9862 |  

- Ensemble CV 0.9889

### Details
- CV strategy
    - MultilabelStratifiedKFold(n_splits=5)
- 300 Epochs
- lr & batch_size
    - lr=4e-4 & batch_size=1024 for Resnet50
    - lr=5e-4 & batch_size=512 for efficientnet-b2
- image size
    - 128x128
- Augmentations
    - ShiftScaleRotate
    - RandomMorph
    - GridDistortion
    - Cutout 
- Cutmix & Mixup (ALPHA=0.4 was better than ALPHA=1)
    - p=0.8 for 1~160 epochs
    - p=0.6 for 161~200 epochs
    - p=0.4 for 201~240 epochs
    - p=0.2 for 241~280 epochs
    - p=0.1 for 281~200 epochs
- Scheduler 
    - ReduceLROnPlateau(factor=0.75, patience=5, eps=1e-6)
- Optimizer
    - Adam
- Loss
    - nn.CrossEntropyLoss() with sample weight
    - loss_weight=\[0.5, 0.25, 0.25, 0.25]
- Generalized Mean Pooling (GeM)
- BalanceSampler

### What didn't work with me
- OHEM
- GridMask (Cutout is enough?)
- RandomAugMix
- Crop black pixels

## Prerequisite
Pull PyTorch image from [NVIDIA GPU CLOUD (NGC)](https://ngc.nvidia.com/)
```
docker login nvcr.io
docker image pull nvcr.io/nvidia/pytorch:20.01-py3
docker run --gpus all -it --ipc=host --name=bengali nvcr.io/nvidia/pytorch:20.01-py3
```

## Usage
```
pip install iterative-stratification
pip install albumentations
```

```
# train model
python train.py
```