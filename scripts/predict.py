from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = "runs/fruit_classifier/weights/best.pt"
TEST_DIR = ROOT / "tests/images"

def predict():
  model = YOLO(MODEL_PATH)

  for image in TEST_DIR.glob("*.jpg"):

    results = model.predict(image)

    r = results[0]
    probs = r.probs

    class_id = probs.top1
    confidence = probs.top1conf
    class_name = r.names[class_id]

    print(f"{image.name} -> {class_name} ({confidence:.2f})")

if __name__ == "__main__":
  predict()

