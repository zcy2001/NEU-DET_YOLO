from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
import time
import shutil
from pathlib import Path

# ============================ 配置 ============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train52" / "weights" / "best.pt"
DEPLOY_DIR = PROJECT_ROOT / "deploy"
DEPLOY_DIR.mkdir(exist_ok=True)
ONNX_PATH = DEPLOY_DIR / "best.onnx"
IMGSZ = 320
# =============================================================

def export_onnx():
    print(f"正在导出 ONNX 模型到默认位置...")
    model = YOLO(str(MODEL_PATH))
    # YOLO默认导出到 best.pt 同级目录
    model.export(format="onnx", imgsz=IMGSZ, half=False)

    # 移动文件到 deploy 目录
    src = MODEL_PATH.parent / "best.onnx"
    if src.exists():
        shutil.move(str(src), str(ONNX_PATH))
        print(f"✅ ONNX 模型已移动到 {ONNX_PATH}")
    else:
        raise FileNotFoundError(f"导出失败，找不到 {src}")

def benchmark_onnx():
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"找不到 ONNX 模型: {ONNX_PATH}")
    print(f"正在加载 ONNX Runtime 模型: {ONNX_PATH}")
    session = ort.InferenceSession(str(ONNX_PATH))
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)  # ONNX使用NCHW格式与float32类型

    # 预热
    for _ in range(10):
        session.run([], {input_name: dummy_input})

    # 测速
    start = time.perf_counter()
    for _ in range(500):
        session.run([], {input_name: dummy_input})
    end = time.perf_counter()
    avg_ms = (end - start) / 500 * 1000
    print(f"🚀 ONNX Runtime (CPU) 平均推理耗时: {avg_ms:.2f} ms")
    return avg_ms

def benchmark_pytorch():
    """对比原始 PyTorch 模型速度"""
    model = YOLO(str(MODEL_PATH))
    dummy = np.random.randn(IMGSZ, IMGSZ, 3).astype(np.uint8)   # YOLO使用HWC格式与uint8类型

    # 预热
    for _ in range(10):
        _ = model(dummy, verbose=False)

    # 测速
    start = time.perf_counter()
    for _ in range(500):
        _ = model(dummy, verbose=False)
    end = time.perf_counter()
    avg_ms = (end - start) / 500 * 1000
    print(f"🔥 PyTorch (CPU) 平均推理耗时: {avg_ms:.2f} ms")
    return avg_ms

if __name__ == "__main__":
    export_onnx()
    benchmark_onnx()
    benchmark_pytorch()