"""FastUMI Retarget Tool CLI。

用法:
  python3 -m fastumi_retarget init --urdf r1_pro.urdf --input ./raw_data/
  python3 -m fastumi_retarget batch --config config.yaml --input ./raw/ --output ./retargeted/
  python3 -m fastumi_retarget evaluate --input ./retargeted/ --config config.yaml
  python3 -m fastumi_retarget merge --input ./retargeted/ --output ./final/ --min-grade PASS
  python3 -m fastumi_retarget auto --urdf r1_pro.urdf --input ./raw/ --output ./final/
"""
import argparse
import os
import sys


def cmd_init(args):
    from .config import scan_subsets, make_default_config, save_config

    print("扫描数据集: {}".format(args.input))
    subsets = scan_subsets(args.input)
    if not subsets:
        print("未找到子集。")
        return 1

    print("\n找到 {} 个子集:".format(len(subsets)))
    for name, info in subsets.items():
        print("  {} ({} eps, {} frames)".format(
            name, info["n_eps"], info["n_frames"]))

    print("\n配置各子集:")
    subsets_config = {}
    for name in subsets:
        mode = input('  "{}": mode? [local/mobile] > '.format(name)).strip()
        if mode not in ("local", "mobile"):
            mode = "mobile"
        hand = input('  "{}": hand? [left/right/both] > '.format(name)).strip()
        if hand not in ("left", "right", "both"):
            hand = "left"
        subsets_config[name] = {"mode": mode, "hand": hand}

    config = make_default_config(args.urdf, subsets_config)
    out_path = args.output or "config.yaml"
    save_config(config, out_path)
    print("\n→ {} 已生成".format(out_path))
    return 0


def cmd_tune(args):
    print("tune: 使用预设 '{}'".format(args.preset))
    print("(交互调参暂未实现, 请直接使用默认预设后 batch)")
    return 0


def cmd_batch(args):
    from .config import load_config
    from .retarget import retarget_batch

    config = load_config(args.config) or {}
    retarget_batch(
        args.input, args.output, config,
        parallel=args.parallel, force=args.force)
    return 0


def cmd_align(args):
    from .config import load_config
    from .align import align_batch

    config = load_config(args.config) or {}
    align_batch(args.input, config)
    return 0


def cmd_evaluate(args):
    from .config import load_config
    from .evaluate import evaluate_batch

    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config) or {}
    evaluate_batch(args.input, config)
    return 0


def cmd_visualize(args):
    print("visualize: viser 可视化暂未实现")
    print("替代: python3 teleop_replay.py --episode <path> --sim")
    return 0


def cmd_merge(args):
    from .merge import merge_dataset

    merge_dataset(args.input, args.output,
                  min_grade=args.min_grade, val_ratio=args.val_ratio)
    return 0


def cmd_auto(args):
    """一键模式: init(默认推断) → batch → evaluate → merge。"""
    from .config import scan_subsets, make_default_config, save_config, load_config
    from .retarget import retarget_batch
    from .evaluate import evaluate_batch
    from .merge import merge_dataset

    config_path = args.config
    if config_path and os.path.exists(config_path):
        print("使用已有 config: {}".format(config_path))
        config = load_config(config_path)
    else:
        print("=== Step 1: init (自动推断) ===")
        subsets = scan_subsets(args.input)
        if not subsets:
            print("未找到子集。")
            return 1

        subsets_config = {}
        for name in subsets:
            mode = "local" if "local" in name.lower() else "mobile"
            hand = "left"
            subsets_config[name] = {"mode": mode, "hand": hand}
            print("  {}: mode={}, hand={}".format(name, mode, hand))

        config = make_default_config(args.urdf, subsets_config)
        config_path = os.path.join(args.output, "config.yaml")
        os.makedirs(args.output, exist_ok=True)
        save_config(config, config_path)

    retarget_dir = os.path.join(args.output, "retargeted")

    print("\n=== Step 3: batch ===")
    retarget_batch(args.input, retarget_dir, config)

    print("\n=== Step 5: evaluate ===")
    evaluate_batch(retarget_dir, config)

    print("\n=== Step 7: merge ===")
    final_dir = os.path.join(args.output, "final")
    merge_dataset(retarget_dir, final_dir, min_grade="PASS")

    print("\n完成! 数据集 → {}".format(final_dir))
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="fastumi-retarget",
        description="FastUMI Retarget Tool: UMI data → full-body joint trajectories")
    subparsers = parser.add_subparsers(dest="command")

    # init
    p = subparsers.add_parser("init", help="扫描数据集, 交互配置, 生成 config.yaml")
    p.add_argument("--urdf", required=True, help="URDF 路径")
    p.add_argument("--input", required=True, help="原始数据集目录")
    p.add_argument("--output", default="config.yaml", help="config.yaml 输出路径")

    # tune
    p = subparsers.add_parser("tune", help="预设对比/交互调参")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--preset", default="balanced")

    # batch
    p = subparsers.add_parser("batch", help="批量 retarget")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--input", required=True, help="原始数据集目录")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--parallel", type=int, default=0)
    p.add_argument("--force", action="store_true", help="强制覆盖已处理的数据")

    # align (Action-Pose Alignment)
    p = subparsers.add_parser("align", help="Action-Pose Alignment (可选, 真机 replay 精度优化)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--input", required=True, help="retargeted 目录")

    # evaluate
    p = subparsers.add_parser("evaluate", help="质量评估, 打 FAIL/PASS/GOOD 标签")
    p.add_argument("--input", required=True, help="retargeted 目录")
    p.add_argument("--config", default="config.yaml")

    # visualize
    p = subparsers.add_parser("visualize", help="可视化 (暂用 mujoco)")
    p.add_argument("--input", required=True)
    p.add_argument("--show", choices=["typical", "fails"], default="typical")
    p.add_argument("--episode", default=None)

    # merge
    p = subparsers.add_parser("merge", help="筛选 + 统一 norm + train/val 划分")
    p.add_argument("--input", required=True, help="retargeted 目录")
    p.add_argument("--output", required=True, help="最终数据集目录")
    p.add_argument("--min-grade", default="PASS", choices=["PASS", "GOOD"])
    p.add_argument("--val-ratio", type=float, default=0.1)

    # auto
    p = subparsers.add_parser("auto", help="一键全流程")
    p.add_argument("--urdf", default=None, help="URDF 路径")
    p.add_argument("--input", required=True, help="原始数据集目录")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--config", default=None, help="已有 config.yaml (跳过 init)")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "tune": cmd_tune,
        "batch": cmd_batch,
        "align": cmd_align,
        "evaluate": cmd_evaluate,
        "visualize": cmd_visualize,
        "merge": cmd_merge,
        "auto": cmd_auto,
    }

    if args.command in commands:
        sys.exit(commands[args.command](args) or 0)
    else:
        parser.print_help()
        sys.exit(1)
