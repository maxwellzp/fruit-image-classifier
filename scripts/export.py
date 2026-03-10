from ultralytics import YOLO
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "runs/fruit_classifier/weights/best.pt"
MODELS_DIR = ROOT / "models"

def export():
    # load my trained model
    model = YOLO(MODEL)

    # Export my trained model to NCNN format
    export_path = model.export(format="ncnn") # creates 'best_ncnn_model'

    # Move exported model
    target = MODELS_DIR / "fruit_classifier_ncnn"
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(export_path, target)

if __name__ == "__main__":
    export()



# (.venv) maksim@maksim-pc:~/fruit-image-classifier$ python3 scripts/export.py 
# Ultralytics 8.4.21 🚀 Python-3.12.3 torch-2.10.0+cu128 CPU (Intel Core i5-4440 3.10GHz)
# YOLO26n-cls summary (fused): 47 layers, 1,529,867 parameters, 0 gradients, 3.2 GFLOPs

# PyTorch: starting from '/home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best.pt' with input shape (1, 3, 224, 224) BCHW and output shape(s) (1, 3) (3.0 MB)

# NCNN: starting export with NCNN 1.0.20260114 and PNNX 20260112...
# pnnxparam = /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best_ncnn_model/model.pnnx.param
# pnnxbin = /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best_ncnn_model/model.pnnx.bin
# pnnxpy = /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best_ncnn_model/model_pnnx.py
# pnnxonnx = /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best_ncnn_model/model.pnnx.onnx
# ncnnparam = /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best_ncnn_model/model.ncnn.param
# ncnnbin = /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best_ncnn_model/model.ncnn.bin
# ncnnpy = /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best_ncnn_model/model_ncnn.py
# fp16 = 0
# optlevel = 2
# device = cpu
# inputshape = [1,3,224,224]f32
# inputshape2 = 
# customop = 
# moduleop = 
# get inputshape from traced inputs
# inputshape = [1,3,224,224]f32
# ############# pass_level0
# inline module = torch.nn.modules.linear.Identity
# inline module = ultralytics.nn.modules.block.Attention
# inline module = ultralytics.nn.modules.block.Bottleneck
# inline module = ultralytics.nn.modules.block.C2PSA
# inline module = ultralytics.nn.modules.block.C3k
# inline module = ultralytics.nn.modules.block.C3k2
# inline module = ultralytics.nn.modules.block.PSABlock
# inline module = ultralytics.nn.modules.conv.Conv
# inline module = ultralytics.nn.modules.head.Classify
# inline module = torch.nn.modules.linear.Identity
# inline module = ultralytics.nn.modules.block.Attention
# inline module = ultralytics.nn.modules.block.Bottleneck
# inline module = ultralytics.nn.modules.block.C2PSA
# inline module = ultralytics.nn.modules.block.C3k
# inline module = ultralytics.nn.modules.block.C3k2
# inline module = ultralytics.nn.modules.block.PSABlock
# inline module = ultralytics.nn.modules.conv.Conv
# inline module = ultralytics.nn.modules.head.Classify

# ----------------

# ############# pass_level1
# ############# pass_level2
# ############# pass_level3
# ############# pass_level4
# ############# pass_level5
# ############# pass_ncnn
# insert_reshape_global_pooling_forward torch.flatten_14 113
# NCNN: export success ✅ 1.5s, saved as '/home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best_ncnn_model' (5.9 MB)

# Export complete (1.6s)
# Results saved to /home/maksim/fruit-image-classifier/runs/fruit_classifier/weights
# Predict:         yolo predict task=classify model=/home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best_ncnn_model imgsz=224 
# Validate:        yolo val task=classify model=/home/maksim/fruit-image-classifier/runs/fruit_classifier/weights/best_ncnn_model imgsz=224 data=/home/maksim/fruit-image-classifier/dataset  
# Visualize:       https://netron.app
# (.venv) maksim@maksim-pc:~/fruit-image-classifier$ 

