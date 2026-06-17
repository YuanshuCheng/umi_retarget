"""合并数据集: 筛选 + 统一 normalization + train/val 划分。"""
import json
import os
import shutil

import numpy as np
import h5py

from .utils import find_datasets


GRADE_ORDER = {"FAIL": 0, "PASS": 1, "GOOD": 2}


def merge_dataset(input_dir, output_dir, min_grade="PASS", val_ratio=0.1):
    """筛选 + 拷贝 + 统一 norm + train/val 划分。"""
    files = find_datasets(input_dir)
    if not files:
        print("未找到 dataset.hdf5: {}".format(input_dir))
        return

    min_level = GRADE_ORDER.get(min_grade, 1)

    selected = []
    rejected = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            grade = str(f.attrs.get("grade", "PASS"))
        if GRADE_ORDER.get(grade, 1) >= min_level:
            selected.append((fp, grade))
        else:
            rejected.append((fp, grade))

    print("筛选: {}/{} 通过 ({} 排除)".format(
        len(selected), len(files), len(rejected)))

    if not selected:
        print("没有通过筛选的数据。")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. 拷贝到输出目录 (保持原结构)
    for fp, grade in selected:
        rel = os.path.relpath(fp, input_dir)
        dst = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(fp, dst)

    # 2. 统一 normalization
    all_actions = []
    all_base = []
    for fp, _ in selected:
        with h5py.File(fp, "r") as f:
            all_actions.append(f["data/demo_0/actions"][:])
            if "data/demo_0/base_actions" in f:
                all_base.append(f["data/demo_0/base_actions"][:])

    all_act = np.concatenate(all_actions, axis=0)
    norm = {
        "action_mean": all_act.mean(axis=0).tolist(),
        "action_std": all_act.std(axis=0).tolist(),
        "action_min": all_act.min(axis=0).tolist(),
        "action_max": all_act.max(axis=0).tolist(),
    }
    if all_base:
        all_b = np.concatenate(all_base, axis=0)
        norm["base_mean"] = all_b.mean(axis=0).tolist()
        norm["base_std"] = all_b.std(axis=0).tolist()
        norm["base_min"] = all_b.min(axis=0).tolist()
        norm["base_max"] = all_b.max(axis=0).tolist()

    # 更新每个 HDF5 的 normalization
    for fp, _ in selected:
        rel = os.path.relpath(fp, input_dir)
        dst = os.path.join(output_dir, rel)
        with h5py.File(dst, "a") as f:
            if "normalization" in f:
                del f["normalization"]
            ng = f.create_group("normalization")
            for k, v in norm.items():
                ng.create_dataset(k, data=np.array(v))

    # 3. train/val 分层划分
    subsets = {}
    for fp, _ in selected:
        rel = os.path.relpath(fp, input_dir)
        subset = rel.split(os.sep)[0]
        subsets.setdefault(subset, []).append(rel)

    train_list, val_list = [], []
    for s in sorted(subsets.keys()):
        eps = subsets[s]
        n_val = max(1, int(len(eps) * val_ratio))
        val_list.extend(eps[-n_val:])
        train_list.extend(eps[:-n_val])

    # 4. 输出文件
    norm_path = os.path.join(output_dir, "normalization.json")
    with open(norm_path, "w") as f:
        json.dump(norm, f, indent=2)

    train_path = os.path.join(output_dir, "train_episodes.txt")
    with open(train_path, "w") as f:
        f.write("\n".join(train_list) + "\n")

    val_path = os.path.join(output_dir, "val_episodes.txt")
    with open(val_path, "w") as f:
        f.write("\n".join(val_list) + "\n")

    # 5. merge_report
    n_good = sum(1 for _, g in selected if g == "GOOD")
    n_pass = sum(1 for _, g in selected if g == "PASS")
    total_frames = sum(len(a) for a in all_actions)

    report = (
        "Merge Report\n"
        "============\n"
        "总计: {}/{} episodes ({} FAIL 已排除)\n"
        "GOOD: {}, PASS: {}\n"
        "总帧数: {}\n"
        "train: {} eps, val: {} eps\n"
    ).format(
        len(selected), len(files), len(rejected),
        n_good, n_pass, total_frames,
        len(train_list), len(val_list))

    report_path = os.path.join(output_dir, "merge_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print("→ {}".format(output_dir))
