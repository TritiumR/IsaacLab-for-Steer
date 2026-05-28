# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pen import mdp
from isaaclab_tasks.manager_based.manipulation.pen.mdp import pen_events

from . import pen_joint_pos_visuomotor_env_cfg


_POINTCLOUD_MASK_DATA_TYPE = "instance_id_segmentation_fast"
_POINTCLOUD_MASK_ROOTS = ("pen", "pen_holder001", "Robotiq_2F_85")


def _configure_pointcloud_table_cameras(env_cfg):
    """Augment the original table camera with depth + instance segmentation, and add a mirrored capture camera."""
    table_cam = env_cfg.scene.table_cam
    table_cam.data_types = ["rgb", "distance_to_image_plane", _POINTCLOUD_MASK_DATA_TYPE]
    table_cam.colorize_instance_id_segmentation = False

    pos = table_cam.offset.pos
    rot = table_cam.offset.rot
    env_cfg.scene.table_cam_mirror = CameraCfg(
        prim_path=str(table_cam.prim_path).replace("table_cam", "table_cam_mirror"),
        height=table_cam.height,
        width=table_cam.width,
        data_types=["rgb", "distance_to_image_plane", _POINTCLOUD_MASK_DATA_TYPE],
        colorize_instance_id_segmentation=False,
        spawn=table_cam.spawn,
        offset=CameraCfg.OffsetCfg(
            pos=(pos[0], -pos[1], pos[2]),
            rot=(-rot[0], rot[1], -rot[2], rot[3]),
            convention=table_cam.offset.convention,
        ),
    )


def _configure_pointcloud_pen_assets(env_cfg):
    """Pointcloud-specific tweaks: double pen-holder x/y scale and rebind pen + holder to the desk's wood material."""
    holder_spawn = env_cfg.scene.pen_holder001.spawn
    sx, sy, sz = holder_spawn.scale
    holder_spawn.scale = (sx * 2.0, sy * 2.0, sz)

    env_cfg.events.retexture_with_table_material = EventTerm(
        func=pen_events.retexture_with_table_material,
        mode="prestartup",
        params={
            "asset_cfgs": [SceneEntityCfg("pen"), SceneEntityCfg("pen_holder001")],
            "table_prim_relpath": "interactive_smalllivingroom/model_table_1",
            "material_subpath": "materials/mat_2086495",
        },
    )


@configclass
class ObservationsCfg(pen_joint_pos_visuomotor_env_cfg.ObservationsCfg):
    """Observation specifications for the point-cloud pen MDP."""

    @configclass
    class PolicyCfg(pen_joint_pos_visuomotor_env_cfg.ObservationsCfg.PolicyCfg):
        point_positions = ObsTerm(
            func=mdp.merged_rgbd_point_cloud_positions,
            params={
                "sensor_names": ("table_cam", "table_cam_mirror"),
                "num_points": 1024,
                "normalize_color": False,
                "segmentation_data_type": _POINTCLOUD_MASK_DATA_TYPE,
                "include_prim_path_roots": _POINTCLOUD_MASK_ROOTS,
            },
        )
        point_color = ObsTerm(
            func=mdp.merged_rgbd_point_cloud_color,
            params={
                "sensor_names": ("table_cam", "table_cam_mirror"),
                "num_points": 1024,
                "normalize_color": False,
                "segmentation_data_type": _POINTCLOUD_MASK_DATA_TYPE,
                "include_prim_path_roots": _POINTCLOUD_MASK_ROOTS,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class DroidPenJointPosPointCloudEnvCfg(
    pen_joint_pos_visuomotor_env_cfg.DroidPenJointPosVisuomotorEnvCfg
):
    """Configuration for the pen task with Droid robot using joint position control and point cloud observations."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_pointcloud_table_cameras(self)
        _configure_pointcloud_pen_assets(self)
