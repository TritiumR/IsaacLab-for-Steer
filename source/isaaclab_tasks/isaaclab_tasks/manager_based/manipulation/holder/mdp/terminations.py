# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

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
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
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
    mug: RigidObject = env.scene[mug_cfg.name]
    holder: RigidObject = env.scene[holder_cfg.name]

    pos_diff = mug.data.root_pos_w - holder.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    z_dist = torch.abs(mug.data.root_pos_w[:, 2] - desired_z)
    on_holder = torch.logical_and(xy_dist <= holder_xy_threshold, z_dist <= z_threshold)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        gripper_open = _gripper_is_open(env, robot, atol=atol, rtol=rtol)
        on_holder = torch.logical_and(on_holder, gripper_open)

    return on_holder


def task_done_holder_stable(
    env: ManagerBasedRLEnv,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("holder"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    holder_xy_threshold: float | None = None,
    min_mug_height: float = 0.36,
    stable_duration_s: float = 1.0,
    max_linear_velocity: float | None = None,
    max_angular_velocity: float | None = None,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Return true after the mug is high enough and released for a consecutive duration."""
    counter_name = "_task_done_holder_stable_success_steps"
    last_step_name = "_task_done_holder_stable_last_step"
    last_result_name = "_task_done_holder_stable_last_result"

    if not hasattr(env, counter_name):
        setattr(env, counter_name, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
        setattr(env, last_step_name, -1)
        setattr(env, last_result_name, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

    current_step = getattr(env, "common_step_counter", -1)
    if getattr(env, last_step_name) == current_step:
        return getattr(env, last_result_name).clone()

    mug: RigidObject = env.scene[mug_cfg.name]
    hanging = mug.data.root_pos_w[:, 2] >= min_mug_height

    if holder_xy_threshold is not None:
        holder: RigidObject = env.scene[holder_cfg.name]
        pos_diff = mug.data.root_pos_w - holder.data.root_pos_w
        xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
        hanging = torch.logical_and(hanging, xy_dist <= holder_xy_threshold)

    if max_linear_velocity is not None:
        lin_vel = torch.linalg.vector_norm(mug.data.root_lin_vel_w, dim=1)
        hanging = torch.logical_and(hanging, lin_vel <= max_linear_velocity)
    if max_angular_velocity is not None:
        ang_vel = torch.linalg.vector_norm(mug.data.root_ang_vel_w, dim=1)
        hanging = torch.logical_and(hanging, ang_vel <= max_angular_velocity)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        hanging = torch.logical_and(hanging, _gripper_is_open(env, robot, atol=atol, rtol=rtol))

    success_steps = getattr(env, counter_name)
    episode_start = env.episode_length_buf <= 1
    success_steps = torch.where(episode_start, torch.zeros_like(success_steps), success_steps)
    success_steps = torch.where(hanging, success_steps + 1, torch.zeros_like(success_steps))
    setattr(env, counter_name, success_steps)

    required_steps = max(1, math.ceil(stable_duration_s / env.step_dt))
    success = success_steps >= required_steps
    setattr(env, last_step_name, current_step)
    setattr(env, last_result_name, success.clone())
    return success


def task_done_holder_released(
    env: ManagerBasedRLEnv,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("holder"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    min_mug_height: float = 0.36,
    holder_xy_threshold: float | None = 0.18,
    min_gripper_mug_distance: float = 0.12,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Stateless success check for annotation: mug is hanging and released."""
    mug: RigidObject = env.scene[mug_cfg.name]
    success = mug.data.root_pos_w[:, 2] >= min_mug_height

    if holder_xy_threshold is not None:
        holder: RigidObject = env.scene[holder_cfg.name]
        pos_diff = mug.data.root_pos_w - holder.data.root_pos_w
        xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
        success = torch.logical_and(success, xy_dist <= holder_xy_threshold)

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        success = torch.logical_and(success, _gripper_is_open(env, robot, atol=atol, rtol=rtol))

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_to_mug_dist = torch.linalg.vector_norm(ee_pos_w - mug.data.root_pos_w, dim=1)
    success = torch.logical_and(success, ee_to_mug_dist >= min_gripper_mug_distance)

    return success
