"""
YOLOv8 训练脚本 - 遵循 YOLO 原生行为
用法：
    python train.py                                                                         # 使用脚本内默认参数
    python train.py --epochs 200 --lr0 0.0005                                               # 覆盖任意参数
    python train.py --model project_root/runs/detect/train_last/weights/last.pt --resume    # 恢复训练（手动指定）
    python train.py --hyp project_root/runs/detect/tune/best_hyperparameters.yaml           # 从 YAML 文件加载超参数
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "data" / "yolo_format" / "data.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8 训练")

    # 基础配置
    basic = parser.add_argument_group("基础参数")
    basic.add_argument("--model", type=str, default="yolov8n.pt", help="模型路径 (.pt 或 .yaml)")
    basic.add_argument("--data", type=str, default=str(DEFAULT_DATA_CONFIG), help="数据集配置")
    basic.add_argument("--hyp", type=str, default=None, help="选择超参数配置 （YAML 文件路径）")
    basic.add_argument("--epochs", type=int, default=100, help="训练轮数")
    basic.add_argument("--batch", type=int, default=16, help="批大小")
    basic.add_argument("--imgsz", type=int, default=320, help="输入图像尺寸（针对 NEU-DET 优化）")
    basic.add_argument("--device", type=str, default="cuda", help="设备，留空自动选择")
    basic.add_argument("--workers", type=int, default=0, help="数据加载线程数")
    basic.add_argument("--project", type=str, default=None, help="输出根目录（默认 runs/train）")
    basic.add_argument("--name", type=str, default=None, help="实验名称")
    basic.add_argument("--exist_ok", action="store_true", default=False, help="允许覆盖已有目录")
    basic.add_argument("--pretrained", action="store_true", default=True, help="使用预训练权重")
    basic.add_argument("--resume", action="store_true", default=False, help="恢复训练（需同时指定 --model 为 last.pt）")
    basic.add_argument("--seed", type=int, default=0, help="随机种子")
    basic.add_argument("--patience", type=int, default=100, help="早停耐心值")
    basic.add_argument("--save_period", type=int, default=-1, help="每 N 轮保存一次")
    basic.add_argument("--verbose", action="store_true", default=False, help="打印详细日志")

    # 优化器
    optim = parser.add_argument_group("优化器")
    optim.add_argument("--optimizer", type=str, default="AdamW", choices=["SGD", "Adam", "AdamW", "auto"])
    optim.add_argument("--lr0", type=float, default=0.001, help="初始学习率")
    optim.add_argument("--lrf", type=float, default=0.01, help="最终学习率因子")
    optim.add_argument("--momentum", type=float, default=0.937, help="SGD 动量")
    optim.add_argument("--weight_decay", type=float, default=0.0005, help="权重衰减")
    optim.add_argument("--warmup_epochs", type=float, default=3.0, help="预热轮数")
    optim.add_argument("--warmup_momentum", type=float, default=0.8, help="预热初始动量")
    optim.add_argument("--box", type=float, default=7.5, help="box 损失权重")
    optim.add_argument("--cls", type=float, default=0.5, help="分类损失权重")
    optim.add_argument("--dfl", type=float, default=1.5, help="DFL 损失权重")
    optim.add_argument("--label_smoothing", type=float, default=0.0, help="标签平滑")

    # 数据增强
    aug = parser.add_argument_group("数据增强")
    aug.add_argument("--hsv_h", type=float, default=0.015, help="色调增强")
    aug.add_argument("--hsv_s", type=float, default=0.7, help="饱和度增强")
    aug.add_argument("--hsv_v", type=float, default=0.4, help="明度增强")
    aug.add_argument("--degrees", type=float, default=0.0, help="随机旋转")
    aug.add_argument("--translate", type=float, default=0.1, help="随机平移")
    aug.add_argument("--scale", type=float, default=0.5, help="随机缩放")
    aug.add_argument("--shear", type=float, default=0.0, help="随机剪切")
    aug.add_argument("--perspective", type=float, default=0.0, help="透视变换")
    aug.add_argument("--flipud", type=float, default=0.0, help="上下翻转概率")
    aug.add_argument("--fliplr", type=float, default=0.5, help="左右翻转概率")
    aug.add_argument("--mosaic", type=float, default=1.0, help="马赛克概率")
    aug.add_argument("--mixup", type=float, default=0.0, help="MixUp 概率")
    aug.add_argument("--copy_paste", type=float, default=0.0, help="Copy-Paste 概率")
    aug.add_argument("--auto_augment", type=str, default="randaugment", help="自动增强策略")
    aug.add_argument("--erasing", type=float, default=0.4, help="随机擦除概率")

    # 模型架构
    arch = parser.add_argument_group("模型架构")
    arch.add_argument("--freeze", type=int, default=0, help="冻结前 N 层")
    arch.add_argument("--dropout", type=float, default=0.0, help="Dropout 率")

    # 日志与缓存
    log = parser.add_argument_group("日志与缓存")
    log.add_argument("--cache", type=str, default="False", choices=["ram", "disk", "False"], help="图像缓存")
    log.add_argument("--plots", action="store_true", default=True, help="生成训练图表")
    log.add_argument("--fraction", type=float, default=1.0, help="数据集使用比例")
    log.add_argument("--profile", action="store_true", help="PyTorch 性能分析")

    return parser


def load_hyperparams(hyp_path: Optional[str]) -> Dict[str, Any]:
    if not hyp_path:
        return {}
    with open(hyp_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_kwargs(args: argparse.Namespace, hyp_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 合并超参数
    cli_dict = {k: v for k, v in vars(args).items() if v is not None and k != "hyp"}
    cli_dict.update(hyp_dict)

    # 只保留 YOLO 官方训练参数（DEFAULT_CFG_DICT 的键）
    filtered = {k: v for k, v in cli_dict.items() if k in DEFAULT_CFG_DICT}

    # 处理 cache 参数的类型转换
    if "cache" in filtered and filtered["cache"] == "False":
        filtered["cache"] = False

    return filtered


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    hyp_dict = load_hyperparams(args.hyp)
    train_kwargs = build_train_kwargs(args, hyp_dict)

    model = YOLO(args.model)
    print(f"🚀 训练参数: {train_kwargs}")
    model.train(**train_kwargs)
    print("✅ 训练完成")


if __name__ == "__main__":
    main()