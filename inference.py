"""
YOLOv8 推理脚本 - 在测试集上运行模型并保存可视化结果
用法：
    python inference.py                         # 使用脚本内默认路径和参数
"""

from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
model_path = PROJECT_ROOT / "runs" / "detect" / "train52" / "weights" / "best.pt"
source_dir = PROJECT_ROOT / "data" / "yolo_format" / "test" / "images"
save_dir = PROJECT_ROOT / "demo"

model = YOLO(model_path)

results = model.predict(
    source=str(source_dir),
    save=True,
    project=str(save_dir),
    name="test_predict",
    exist_ok=True,
    conf=0.15,                
    iou=0.3,
)

print(f"预测完成，图片保存在: {save_dir / 'test_predict'}")