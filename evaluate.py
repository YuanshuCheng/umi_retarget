"""质量评估: 对 retarget 后的数据打 FAIL/PASS/GOOD 标签。"""
import os

import numpy as np
import h5py

from .utils import find_datasets


def evaluate_episode(hdf5_path, eval_config=None):
    """对单条 episode 评估, 返回 {grade, score, scores, fail_reasons}。"""
    if eval_config is None:
        eval_config = {}

    fail_limit_ratio = eval_config.get("fail_joint_limit_ratio", 0.1)
    fail_stuck_sec = eval_config.get("fail_stuck_seconds", 3.0)
    fail_min_dur = eval_config.get("fail_min_duration", 2.0)
    fail_spin_deg = eval_config.get("fail_chassis_spin_deg", 360.0)
    good_threshold = eval_config.get("good_threshold", 0.7)

    with h5py.File(hdf5_path, "r") as f:
        demo = f["data/demo_0"]
        jp = demo["obs/joint_positions"][:]
        N = len(jp)
        freq = 30.0

        overlap_l = demo["quality/tracking_left_overlap"][:] if "quality/tracking_left_overlap" in demo else None
        overlap_r = demo["quality/tracking_right_overlap"][:] if "quality/tracking_right_overlap" in demo else None
        kf_mask = demo["quality/keyframe_mask"][:] if "quality/keyframe_mask" in demo else None
        base = demo["obs/base_position"][:] if "obs/base_position" in demo else None

        physics_pass = float(demo["quality/physics_pass_ratio"][()]) if "quality/physics_pass_ratio" in demo else 1.0

    overlap = overlap_l if overlap_l is not None else overlap_r
    if overlap is None:
        overlap = np.zeros(N)

    # === FAIL 判定 ===
    fail_reasons = []

    if N < int(fail_min_dur * freq):
        fail_reasons.append("过短 ({:.1f}s < {:.1f}s)".format(N / freq, fail_min_dur))

    if base is not None and base.shape[1] >= 3:
        yaw_total = np.degrees(abs(base[-1, 2] - base[0, 2]))
        if yaw_total > fail_spin_deg:
            fail_reasons.append("底盘打转 (yaw={:.0f}°)".format(yaw_total))

    joint_diff = np.abs(np.diff(jp[:, :18], axis=0))
    max_still = 0
    count = 0
    for t in range(len(joint_diff)):
        if joint_diff[t].max() < 0.001:
            count += 1
            max_still = max(max_still, count)
        else:
            count = 0
    stuck_sec = max_still / freq
    if stuck_sec > fail_stuck_sec:
        fail_reasons.append("卡住 {:.1f}s".format(stuck_sec))

    if physics_pass < (1.0 - fail_limit_ratio):
        fail_reasons.append("物理不通过 ({:.0f}%)".format(physics_pass * 100))

    if fail_reasons:
        return {
            "grade": "FAIL",
            "score": 0.0,
            "scores": {},
            "fail_reasons": fail_reasons,
            "n_frames": N,
        }

    # === PASS vs GOOD 评分 (绝对阈值) ===
    scores = {}

    overlap_mean_cm = float(overlap.mean() * 100)
    scores["tracking"] = 1.0 if overlap_mean_cm < 2 else (0.5 if overlap_mean_cm < 5 else 0.0)

    if kf_mask is not None and kf_mask.sum() > 0:
        kf_overlap_cm = float(overlap[kf_mask.astype(bool)].mean() * 100)
        scores["keyframe"] = 1.0 if kf_overlap_cm < 1 else (0.5 if kf_overlap_cm < 3 else 0.0)
    else:
        scores["keyframe"] = 0.5

    dt = 1.0 / freq
    vel = np.diff(jp[:, :18], axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
    scores["smoothness"] = 1.0 if jerk_rms < 50 else (0.5 if jerk_rms < 150 else 0.0)

    torso_pitch = np.degrees(jp[:, 0] + jp[:, 1])
    pitch_range = float(torso_pitch.max() - torso_pitch.min())
    scores["humanlike"] = 1.0 if pitch_range < 30 else (0.5 if pitch_range < 60 else 0.0)

    final_score = float(np.mean(list(scores.values())))
    grade = "GOOD" if final_score >= good_threshold else "PASS"

    return {
        "grade": grade,
        "score": final_score,
        "scores": scores,
        "fail_reasons": [],
        "n_frames": N,
        "overlap_cm": overlap_mean_cm,
        "jerk_rms": jerk_rms,
        "pitch_range": pitch_range,
    }


def evaluate_batch(input_dir, config):
    """批量评估所有 episode, 打标签, 输出汇总。"""
    eval_config = config.get("evaluation", {})
    files = find_datasets(input_dir)

    if not files:
        print("未找到 dataset.hdf5: {}".format(input_dir))
        return []

    results = []
    for fp in files:
        rel = os.path.relpath(fp, input_dir)
        result = evaluate_episode(fp, eval_config)
        result["path"] = fp
        result["rel"] = rel
        parts = rel.split(os.sep)
        result["subset"] = parts[0] if len(parts) > 1 else ""
        results.append(result)

        with h5py.File(fp, "a") as f:
            f.attrs["grade"] = result["grade"]
            f.attrs["quality_score"] = result["score"]

    _print_report(results, input_dir)
    return results


def _print_report(results, input_dir):
    """打印汇总报告。"""
    print("\n" + "=" * 60)
    print("质量评估报告 ({} episodes)".format(len(results)))
    print("=" * 60)

    subsets = sorted(set(r["subset"] for r in results))
    print("\n{:<22} {:>3} {:>5} {:>5} {:>5} {:>6}".format(
        "子集", "N", "GOOD", "PASS", "FAIL", "通过率"))
    print("-" * 50)

    for s in subsets:
        sr = [r for r in results if r["subset"] == s]
        n = len(sr)
        good = sum(1 for r in sr if r["grade"] == "GOOD")
        pas = sum(1 for r in sr if r["grade"] == "PASS")
        fail = sum(1 for r in sr if r["grade"] == "FAIL")
        rate = (good + pas) / max(n, 1) * 100
        print("{:<22} {:>3} {:>5} {:>5} {:>5} {:>5.0f}%".format(
            s, n, good, pas, fail, rate))

    total = len(results)
    tg = sum(1 for r in results if r["grade"] == "GOOD")
    tp = sum(1 for r in results if r["grade"] == "PASS")
    tf = sum(1 for r in results if r["grade"] == "FAIL")
    print("-" * 50)
    print("{:<22} {:>3} {:>5} {:>5} {:>5} {:>5.0f}%".format(
        "总计", total, tg, tp, tf, (tg + tp) / max(total, 1) * 100))

    fails = [r for r in results if r["grade"] == "FAIL"]
    if fails:
        print("\nFAIL 原因:")
        for r in fails:
            print("  {}: {}".format(r["rel"], ", ".join(r["fail_reasons"])))

    report_path = os.path.join(input_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("质量评估: {}/{} 通过\n".format(tg + tp, total))
        for r in results:
            f.write("{}: {} (score={:.2f})\n".format(
                r["rel"], r["grade"], r["score"]))
        if fails:
            f.write("\nFAIL:\n")
            for r in fails:
                f.write("  {}: {}\n".format(r["rel"], ", ".join(r["fail_reasons"])))
    print("\n→ {}".format(report_path))
    print("=" * 60)
