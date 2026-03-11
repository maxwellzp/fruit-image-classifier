from ultralytics import YOLO
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "dataset"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"

def copy_best_model():
    src_model = "runs/fruit_classifier/weights/best.pt"
    dst_model = MODELS / "fruit_classifier.pt"
    shutil.copy(src_model, dst_model)

def train():
  model = YOLO(ROOT / "pretrained/yolo26n-cls.pt")

  model.train(
    data= DATASET,
    epochs=50,
    batch=8,
    imgsz=224,
    device="cpu",
    project=RUNS,
    name="fruit_classifier",
    exist_ok=True,
  )

  print("Training finished!")

if __name__ == "__main__":
  train()
  copy_best_model()

# 50 epochs completed in 0.897 hours.
# Optimizer stripped from /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/last.pt, 3.2MB
# Optimizer stripped from /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best.pt, 3.2MB

# Validating /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best.pt...
# Ultralytics 8.4.21 🚀 Python-3.12.3 torch-2.10.0+cu128 CPU (Intel Core i5-4440 3.10GHz)
# YOLO26n-cls summary (fused): 47 layers, 1,529,867 parameters, 0 gradients, 3.2 GFLOPs
# train: /home/maksim/fruit-image-classifier/dataset/train... found 240 images in 3 classes ✅ 
# val: /home/maksim/fruit-image-classifier/dataset/val... found 60 images in 3 classes ✅ 
# test: None...
#                classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 4/4 2.8s/it 11.2s
#                    all          1          1
# Speed: 0.0ms preprocess, 5.2ms inference, 0.0ms loss, 0.0ms postprocess per image
# Results saved to /home/maksim/fruit-image-classifier/runs/fruit_classifier
# Training finished!
