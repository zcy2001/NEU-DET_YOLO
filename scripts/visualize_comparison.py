"""
visualize_comparison.py - 生成真实标注 vs 模型预测的对比图，并排保存，用于直观评估检测质量。
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ============================= 基础配置 =============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train52" / "weights" / "best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "data" / "yolo_format" / "test" / "images"
TEST_LABEL_DIR = PROJECT_ROOT / "data" / "yolo_format" / "test" / "labels"
OUTPUT_DIR = PROJECT_ROOT / "demo" / "comparison"

# 类别名称
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# 颜色 (BGR)
GT_COLOR = (0, 255, 0)      # 绿色 - 真实框
PRED_COLOR = (0, 0, 255)    # 红色 - 预测框
# ===================================================================

def load_labels(label_path, img_shape):
    """从 YOLO txt 文件读取标注，转换为像素坐标 (x1, y1, x2, y2, class_id)"""
    h, w = img_shape[:2]
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_c = float(parts[1]) * w
            y_c = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h
            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2)
            y2 = int(y_c + bh / 2)
            boxes.append((x1, y1, x2, y2, class_id))
    return boxes

def draw_boxes(image, boxes, color, label_names=None):
    """在图像上绘制边界框，可选显示类别名"""
    img_copy = image.copy()
    for (x1, y1, x2, y2, cls_id) in boxes:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        if label_names:
            text = label_names[cls_id] if cls_id < len(label_names) else str(cls_id)
            cv2.putText(img_copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"已加载模型: {MODEL_PATH}")

    img_paths = sorted(TEST_IMG_DIR.glob("*.jpg")) + sorted(TEST_IMG_DIR.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"在 {TEST_IMG_DIR} 中没有找到图片")

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning：无法读取 {img_path}")
            continue

        # 加载真实标签
        label_path = TEST_LABEL_DIR / (img_path.stem + ".txt")
        gt_boxes = load_labels(label_path, img.shape)

        # 模型预测
        results = model(img_path, iou=0.3, conf=0.15, verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                pred_boxes.append((x1, y1, x2, y2, cls_id))

        # 分别绘制真实框和预测框
        img_gt = draw_boxes(img, gt_boxes, GT_COLOR, CLASS_NAMES)
        img_pred = draw_boxes(img, pred_boxes, PRED_COLOR, CLASS_NAMES)

        # 并排拼接（宽度相同，高度相同）
        h, w = img.shape[:2]
        combined = np.hstack((img_gt, img_pred))

        # 添加顶部门楣文字
        header = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, "Ground Truth", (w//2 - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(header, "Prediction", (w + w//2 - 70, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        combined_with_header = np.vstack((header, combined))

        # 保存
        out_path = OUTPUT_DIR / f"{img_path.stem}_compare.jpg"
        cv2.imwrite(str(out_path), combined_with_header)
        print(f"已保存对比图: {out_path} (真实框: {len(gt_boxes)}个, 预测框: {len(pred_boxes)}个)")

    print(f"\n所有对比图已生成，保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()