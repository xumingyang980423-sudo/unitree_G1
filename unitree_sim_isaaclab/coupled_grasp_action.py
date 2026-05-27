"""Dual-group grasp action for Inspire hand: 4 fingers move as one unit, thumb opposes."""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Four-finger group: one scalar -> identical normalized closure on all 8 joints.
FOUR_FINGER_JOINT_NAMES = [
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
]

THUMB_JOINT_NAMES = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]

_FINGER_BODY_PREFIXES = ("R_index", "R_middle", "R_ring", "R_pinky")
_FINGER_BODY_SUFFIXES = ("distal", "intermediate", "proximal")
GRASP_ENABLE_DISTANCE = 0.11
GRASP_GATE_STD = 0.025

# Per-joint range (rad) and role within each group when action = +1.
# Intermediate joints follow proximal with a fixed fraction (Inspire coupling).
_FOUR_FINGER_SPEC = {
    "R_index_proximal_joint": (0.0, 1.7, 1.00),
    "R_index_intermediate_joint": (0.0, 1.7, 0.88),
    "R_middle_proximal_joint": (0.0, 1.7, 1.00),
    "R_middle_intermediate_joint": (0.0, 1.7, 0.88),
    "R_pinky_proximal_joint": (0.0, 0.5, 1.00),
    "R_pinky_intermediate_joint": (0.0, 0.5, 0.88),
    "R_ring_proximal_joint": (0.0, 0.5, 1.00),
    "R_ring_intermediate_joint": (0.0, 0.5, 0.88),
}

_THUMB_SPEC = {
    "R_thumb_proximal_yaw_joint": (-0.1, 1.3, 0.90),
    "R_thumb_proximal_pitch_joint": (-0.1, 1.3, 1.00),
    "R_thumb_intermediate_joint": (-0.1, 1.3, 0.90),
    "R_thumb_distal_joint": (-0.1, 1.3, 0.78),
}


class DualGroupGraspAction(ActionTerm):
    """2-D grasp: [four_finger_curl, thumb_curl] with coordinated joint deltas."""

    cfg: DualGroupGraspActionCfg
    _asset: Articulation

    def __init__(self, cfg: DualGroupGraspActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._finger_ids, self._finger_names = self._asset.find_joints(
            cfg.finger_joint_names, preserve_order=True
        )
        self._thumb_ids, self._thumb_names = self._asset.find_joints(
            cfg.thumb_joint_names, preserve_order=True
        )

        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros(
            self.num_envs, len(self._finger_ids) + len(self._thumb_ids), device=self.device
        )

        self._finger_spans, self._finger_roles = self._build_specs(cfg.finger_spec, self._finger_names)
        self._thumb_spans, self._thumb_roles = self._build_specs(cfg.thumb_spec, self._thumb_names)
        self._env = env

    def _hand_to_object_dist(self) -> torch.Tensor:
        """Distance from fingertip centroid to block center."""
        robot = self._asset
        obj = self._env.scene["object"].data.root_pos_w
        tips = []
        for name in robot.data.body_names:
            if not any(name.startswith(p) for p in _FINGER_BODY_PREFIXES):
                continue
            if not any(s in name for s in _FINGER_BODY_SUFFIXES):
                continue
            tips.append(robot.data.body_pos_w[:, robot.data.body_names.index(name)])
        if tips:
            hand = torch.stack(tips, dim=1).mean(dim=1)
        else:
            wrist_idx = robot.data.body_names.index("right_wrist_yaw_link")
            hand = robot.data.body_pos_w[:, wrist_idx]
        return torch.norm(obj - hand, dim=1)

    def _grasp_proximity_gate(self) -> torch.Tensor:
        """0 = far (cannot close), 1 = at grasp distance."""
        dist = self._hand_to_object_dist()
        return torch.clamp(
            (self.cfg.grasp_enable_distance - dist) / self.cfg.grasp_gate_std, 0.0, 1.0
        ).unsqueeze(-1)

    @staticmethod
    def _build_specs(spec: dict[str, tuple[float, float, float]], joint_names: list[str]):
        lo_list, span_list, role_list = [], [], []
        for name in joint_names:
            if name not in spec:
                raise ValueError(f"Missing joint spec for {name}")
            lo, hi, role = spec[name]
            lo_list.append(lo)
            span_list.append(hi - lo)
            role_list.append(role)
        return (
            torch.tensor(span_list, dtype=torch.float32),
            torch.tensor(role_list, dtype=torch.float32),
        )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        finger_a = self._raw_actions[:, 0:1]
        thumb_a = self._raw_actions[:, 1:2]

        # Stage 2 gate: only allow closing when hand is near the block (user pipeline).
        gate = self._grasp_proximity_gate()
        finger_a = torch.clamp(finger_a, max=0.0) + torch.clamp(finger_a, min=0.0) * gate
        thumb_a = torch.clamp(thumb_a, max=0.0) + torch.clamp(thumb_a, min=0.0) * gate

        finger_spans = self._finger_spans.to(self.device)
        finger_roles = self._finger_roles.to(self.device)
        thumb_spans = self._thumb_spans.to(self.device)
        thumb_roles = self._thumb_roles.to(self.device)

        finger_delta = finger_a * self.cfg.scale * finger_spans * finger_roles
        thumb_delta = thumb_a * self.cfg.scale * thumb_spans * thumb_roles

        nf = len(self._finger_ids)
        self._processed_actions[:, :nf] = finger_delta
        self._processed_actions[:, nf:] = thumb_delta

    def apply_actions(self):
        all_ids = self._finger_ids + self._thumb_ids
        current = self._asset.data.joint_pos[:, all_ids] + self._processed_actions
        self._asset.set_joint_position_target(current, joint_ids=all_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class DualGroupGraspActionCfg(ActionTermCfg):
    """Four fingers share one curl command; thumb uses a separate curl command."""

    class_type: type[ActionTerm] = DualGroupGraspAction
    finger_joint_names: list[str] = MISSING
    thumb_joint_names: list[str] = MISSING
    finger_spec: dict[str, tuple[float, float, float]] = MISSING
    thumb_spec: dict[str, tuple[float, float, float]] = MISSING
    scale: float = 0.10
    grasp_enable_distance: float = GRASP_ENABLE_DISTANCE
    grasp_gate_std: float = GRASP_GATE_STD
