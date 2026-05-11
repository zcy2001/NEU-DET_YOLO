"""
app.py - Gradio 交互式缺陷检测界面
"""

import time
import gradio as gr
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# ============================ 配置 ============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train52" / "weights" / "best.pt"
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
# =============================================================

model = YOLO(str(MODEL_PATH))

def predict(image):
    """
    输入：PIL Image
    返回：带框的PIL图片及检测信息
    """
    start_time = time.perf_counter()
    results = model(image, conf=0.15, iou=0.3)
    inference_ms = (time.perf_counter() - start_time) * 1000

    plotted = results[0].plot()
    output_image = Image.fromarray(plotted[..., ::-1])   # BGR -> RGB

    detections = results[0].boxes
    lines = [f"⏱️ 推理时间: {inference_ms:.1f} ms"]
    if detections is not None:
        num_det = len(detections)
        lines.append(f"🔍 检测到缺陷数量: {num_det}")
        for i, (xyxy, conf, cls) in enumerate(zip(detections.xyxy, detections.conf, detections.cls)):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            class_name = CLASS_NAMES[int(cls)]
            lines.append(
                f"  {i+1}. {class_name} (置信度: {conf:.2f})  "
                f"框坐标: [{x1}, {y1}, {x2}, {y2}]"
            )
    else:
        lines.append("✅ 未检测到缺陷")

    detail_text = "\n".join(lines)
    return output_image, detail_text

# 构建 Gradio 界面
with gr.Blocks(title="NEU-DET 钢材表面缺陷检测") as demo:
    gr.Markdown("# 🔍 钢材表面缺陷检测系统")
    gr.Markdown("上传一张热轧钢带图片，模型将自动识别 6 类缺陷：crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="输入图片")
            submit_btn = gr.Button("开始检测")
        with gr.Column():
            output_img = gr.Image(type="pil", label="检测结果")
            output_text = gr.Textbox(label="检测明细", lines=6)
    submit_btn.click(fn=predict, inputs=input_img, outputs=[output_img, output_text])
    gr.Examples(
        examples=[
            str(PROJECT_ROOT / "data" / "yolo_format" / "test" / "images" / "crazing_6.jpg"),
            str(PROJECT_ROOT / "data" / "yolo_format" / "test" / "images" / "inclusion_222.jpg"),
            str(PROJECT_ROOT / "data" / "yolo_format" / "test" / "images" / "patches_10.jpg"),
        ],
        inputs=input_img
    )

if __name__ == "__main__":
    demo.launch(share=False)   # share=True 会生成公网链接，但需要外网访问