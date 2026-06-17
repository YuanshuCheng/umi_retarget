"""共用工具函数。"""
import glob
import os


def find_datasets(input_dir):
    """递归查找所有 dataset.hdf5。"""
    return sorted(glob.glob(
        os.path.join(input_dir, "**", "dataset.hdf5"), recursive=True))


def find_episodes(input_dir):
    """递归查找所有 episode_*.hdf5 (原始数据)。"""
    return sorted(glob.glob(
        os.path.join(input_dir, "**", "episode_*.hdf5"), recursive=True))


def rel_path(filepath, base_dir):
    """返回相对路径。"""
    return os.path.relpath(filepath, base_dir)
