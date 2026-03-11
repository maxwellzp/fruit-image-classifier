# Fruit Image Classifier (YOLOv26 Classification)

Simple computer vision project for fruit image classification using the latest YOLO classification model.

## Dataset

For this project I created my own dataset:

- 100 apple photos
- 100 banana photos
- 100 mandarin photos

Dataset structure:

```
dataset/
   train/
      apple/
      banana/
      mandarin/
   val/
      apple/
      banana/
      mandarin/
```

## Training

Train model:

```
python3 scripts/train.py
```

## Prediction

Run inference:

```
python3 scripts/predict.py
```

## Model export

Export model:

```
python3 scripts/export.py
```
