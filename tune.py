"""
YOLOv8 超参数调优脚本（进化算法和 Ray Tune）
用法：
    python tune.py                               # 使用 YOLO 默认搜索空间，进化算法
    python tune.py --hyp search_space.yaml       # 使用自定义搜索空间
    python tune.py --use_ray --iterations 100    # 启用 Ray Tune 并行调优
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "data" / "yolo_format" / "data.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8 超参数调优")

    # 基础参数
    basic = parser.add_argument_group("基础参数")
    basic.add_argument("--model", type=str, default="yolov8n.pt", help="模型路径 (.pt 或 .yaml)")
    basic.add_argument("--data", type=str, default=str(DEFAULT_DATA_CONFIG), help="数据集配置")
    basic.add_argument("--epochs", type=int, default=30, help="每次试验的训练轮数（调优）")
    basic.add_argument("--batch", type=int, default=16, help="批大小")
    basic.add_argument("--imgsz", type=int, default=320, help="输入图像尺寸")
    basic.add_argument("--device", type=str, default="cuda", help="计算设备（留空自动选择）")
    basic.add_argument("--workers", type=int, default=0, help="数据加载线程数")
    basic.add_argument("--project", type=str, default=None, help="输出根目录（默认 runs/tune）")
    basic.add_argument("--name", type=str, default=None, help="调优实验名称")
    basic.add_argument("--seed", type=int, default=0, help="随机种子")
    basic.add_argument("--verbose", action="store_true", default=False, help="打印详细日志")

    # 超参数搜索空间
    basic.add_argument("--hyp", type=str, default=None,
                       help="超参数搜索空间 YAML 文件（定义待调优的参数及其取值范围，不提供则使用 YOLO 默认）")

    # 调优控制
    tune_ctrl = parser.add_argument_group("调优控制")
    tune_ctrl.add_argument("--iterations", type=int, default=50, help="调优迭代次数（试验组数）")
    tune_ctrl.add_argument("--use_ray", action="store_true", help="启用 Ray Tune 并行调优（否则使用进化算法）")
    tune_ctrl.add_argument("--plots", action="store_true", default=True, help="生成调优过程图表")
    tune_ctrl.add_argument("--tune_grace_period", type=int, default=10,
                           help="ASHA 调度器的早停前轮数（仅 use_ray=True 时有效）")
    tune_ctrl.add_argument("--tune_reduction_factor", type=int, default=4,
                           help="ASHA 调度器的缩减因子（仅 use_ray=True 时有效）")
    tune_ctrl.add_argument("--tune_resources", type=str, default="1",
                           help="每个试验使用的 GPU 数量，如 '1'（仅 use_ray=True 时有效）")
    tune_ctrl.add_argument("--tune_scheduler", type=str, default="asha",
                           choices=["asha", "median", "hyperband"],
                           help="Ray Tune 调度器（仅 use_ray=True 时有效）")

    return parser


def build_tune_kwargs(args: argparse.Namespace) -> dict:
    """构建传递给 model.tune() 的参数字典"""
    kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "verbose": args.verbose,
        "iterations": args.iterations,
        "use_ray": args.use_ray,
        "plots": args.plots,
        "save": True,
    }

    if args.hyp:
        kwargs["hyp"] = args.hyp

    if args.use_ray:
        kwargs["grace_period"] = args.tune_grace_period
        kwargs["reduction_factor"] = args.tune_reduction_factor
        kwargs["resources_per_trial"] = {"gpu": int(args.tune_resources)}
        kwargs["scheduler"] = args.tune_scheduler

    return kwargs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tune_kwargs = build_tune_kwargs(args)

    model = YOLO(args.model)
    print(f"🔧 开始超参数调优: {args.iterations} 次试验，每次 {args.epochs} 轮")
    print(f"   使用 {'Ray Tune' if args.use_ray else '遗传进化算法'} 模式")
    if args.hyp:
        print(f"   自定义搜索空间: {args.hyp}")

    model.tune(**tune_kwargs)

    print("\n✅ 调优完成！最佳超参数文件保存在输出目录下的 best_hyperparameters.yaml 中。")
    print("   请根据实际路径，在训练时使用以下命令加载超参数：")
    print("   python train.py --hyp path_to_best_hyperparameters.yaml")


if __name__ == "__main__":
    main()