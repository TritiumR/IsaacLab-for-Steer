# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.manipulation.plate.mdp.terminations import root_height_below_minimum  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _gripper_is_open(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Check if both gripper joints are close to the configured open value."""
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"
    gripper_open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)

    gripper_1_open = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[0]],
        gripper_open_val,
        atol=atol,
        rtol=rtol,
    )
    gripper_2_open = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[1]],
        gripper_open_val,
        atol=atol,
        rtol=rtol,
    )
    return torch.logical_and(gripper_1_open, gripper_2_open)


def task_done_holder(
    env: ManagerBasedRLEnv,
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("holder"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    holder_xy_threshold: float = 0.15,
    desired_z: float = 0.40,
    z_threshold: float = 0.05,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Cup is placed on the holder and released."""
    cup: RigidObject = env.scene[cup_cfg.name]
    holder: RigidObject = env.scene[holder_cfg.name]

    pos_diff = cup.data.root_pos_w - holder.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    z_dist = torch.abs(cup.data.root_pos_w[:, 2] - desired_z)
    on_holder = torch.logical_and(xy_dist <= holder_xy_threshold, z_dist <= z_threshold)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        gripper_open = _gripper_is_open(env, robot, atol=atol, rtol=rtol)
        on_holder = torch.logical_and(on_holder, gripper_open)

    return on_holder
