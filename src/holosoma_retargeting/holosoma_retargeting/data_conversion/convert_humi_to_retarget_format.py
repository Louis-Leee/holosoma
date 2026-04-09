#!/usr/bin/env python3
"""Convert HuMI LeRobot parquet datasets to LAFAN-compatible qpos NPZ format.

HuMI datasets already contain IK-solved G1-29DOF joint positions. This script
converts them into the same NPZ format produced by the retargeting pipeline,
so downstream tools (viser_player.py, convert_data_format_mj.py) work unchanged.

Usage:
    cd src/holosoma_retargeting/holosoma_retargeting
    python data_conversion/convert_humi_to_retarget_format.py \
        --humi-dir ../../../../data/humi/HuMI-Proposal \
        --save-dir demo_results/g1/robot_only/humi
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ConvertHumiConfig:
    """Configuration for HuMI → retarget-format conversion."""

    humi_dir: str
    """Path to HuMI dataset directory (containing data/, meta/)."""

    save_dir: str = "demo_results/g1/robot_only/humi"
    """Output directory for converted NPZ files."""

    dataset_name: str | None = None
    """Override dataset name. Default: infer from humi_dir (e.g. HuMI-Proposal → proposal)."""

    fps_override: int | None = None
    """Override FPS. Default: read from meta/info.json."""


def axis_angle_to_quat_wxyz(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (T, 3) to quaternion wxyz (T, 4)."""
    r = Rotation.from_rotvec(aa)
    q_xyzw = r.as_quat()  # scipy returns [x, y, z, w]
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]  # reorder to [w, x, y, z]
    return q_wxyz


def infer_dataset_name(humi_dir: str) -> str:
    """Infer short dataset name from directory path.

    HuMI-Proposal → proposal
    HuMI-Walk-Clean-Table → walk_clean_table
    HuMI-Toss → toss
    """
    dirname = Path(humi_dir).name
    # Strip "HuMI-" prefix
    name = re.sub(r"^HuMI-", "", dirname, flags=re.IGNORECASE)
    # Convert kebab-case to snake_case and lowercase
    name = name.replace("-", "_").lower()
    return name


def read_fps_from_info(humi_dir: str) -> int:
    """Read FPS from meta/info.json."""
    info_path = Path(humi_dir) / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    return int(info["fps"])


def load_all_parquets(humi_dir: str) -> pd.DataFrame:
    """Load and concatenate all parquet data files."""
    pattern = str(Path(humi_dir) / "data" / "chunk-*" / "file-*.parquet")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found matching: {pattern}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def convert_episode(episode_df: pd.DataFrame) -> np.ndarray:
    """Convert a single episode DataFrame to qpos array (T, 36).

    qpos layout (MuJoCo order):
        [0:3]   root position (xyz)
        [3:7]   root quaternion (wxyz)
        [7:36]  joint angles (29 DOF)
    """
    # Extract fields
    joint_pos = np.stack(episode_df["joint_pos"].values)  # (T, 29)
    pelvis_pos = np.stack(episode_df["robot2_eef_pos"].values)  # (T, 3)
    pelvis_rot_aa = np.stack(episode_df["robot2_eef_rot_axis_angle"].values)  # (T, 3)

    # Convert pelvis rotation: axis-angle → quaternion wxyz
    pelvis_quat = axis_angle_to_quat_wxyz(pelvis_rot_aa)  # (T, 4)

    # Assemble qpos: [pos(3), quat(4), joints(29)] = (T, 36)
    qpos = np.concatenate([pelvis_pos, pelvis_quat, joint_pos], axis=1)
    return qpos.astype(np.float64)


def main(cfg: ConvertHumiConfig) -> None:
    """Main conversion function."""
    humi_dir = cfg.humi_dir
    save_dir = Path(cfg.save_dir)
    dataset_name = cfg.dataset_name or infer_dataset_name(humi_dir)
    fps = cfg.fps_override or read_fps_from_info(humi_dir)

    print(f"[convert_humi] Dataset: {dataset_name}")
    print(f"[convert_humi] Source: {humi_dir}")
    print(f"[convert_humi] Output: {save_dir}")
    print(f"[convert_humi] FPS: {fps}")

    # Load all parquet data
    df = load_all_parquets(humi_dir)
    episode_ids = sorted(df["episode_index"].unique())
    print(f"[convert_humi] Total frames: {len(df)}, episodes: {len(episode_ids)}")

    # Create output directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert each episode
    for ep_idx in episode_ids:
        episode_df = df[df["episode_index"] == ep_idx].sort_values("frame_index").reset_index(drop=True)
        qpos = convert_episode(episode_df)

        filename = f"{dataset_name}_ep{ep_idx:03d}.npz"
        out_path = save_dir / filename
        np.savez(out_path, qpos=qpos, fps=fps)

        print(f"  Saved {filename}: qpos {qpos.shape}, {qpos.shape[0] / fps:.1f}s")

    print(f"[convert_humi] Done! {len(episode_ids)} episodes saved to {save_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(ConvertHumiConfig)
    main(cfg)
