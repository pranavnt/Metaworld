from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.envs.mujoco.utils import reward_utils
from metaworld.types import InitConfigDict


class SawyerDrawerOpenEnvV2(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.0)
        obj_high = (0.1, 0.9, 0.0)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.9, 0.0], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]


        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

        self.alive_indices = np.array([
            0, 1, 2, 3, 4, 5, 6,
            18, 19, 20, 21, 22, 23, 24,
            36, 37, 38
        ])

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_drawer.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            gripper_error,
            gripped,
            handle_error,
            caging_reward,
            opening_reward,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(handle_error <= 0.03),
            "near_object": float(gripper_error <= 0.03),
            "grasp_success": float(gripped > 0),
            "grasp_reward": caging_reward,
            "in_place_reward": opening_reward,
            "obj_to_target": handle_error,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("objGeom")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.0])

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("drawer_link").xquat

    def reset_model(self) -> npt.NDArray[np.float64]:
        self.hand_init_pos = self.np_random.uniform(
            self.hand_low,
            self.hand_high,
            size=self.hand_low.shape[0]
        )
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec()
        # Set mujoco body to computed position
        self.model.body("drawer").pos = self.obj_init_pos

        # Set _target_pos to current drawer position (closed) minus an offset
        self._target_pos = self.obj_init_pos + np.array(
            [0.0, -0.16 - self.maxDist, 0.09]
        )
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        gripper = obs[:3]
        handle = obs[4:7]

        handle_error = float(np.linalg.norm(handle - self._target_pos))

        reward_for_opening = reward_utils.tolerance(
            handle_error, bounds=(0, 0.02), margin=self.maxDist, sigmoid="long_tail"
        )

        handle_pos_init = self._target_pos + np.array([0.0, self.maxDist, 0.0])
        # Emphasize XY error so that gripper is able to drop down and cage
        # handle without running into it. By doing this, we are assuming
        # that the reward in the Z direction is small enough that the agent
        # will be willing to explore raising a finger above the handle, hook it,
        # and drop back down to re-gain Z reward
        scale = np.array([3.0, 3.0, 1.0])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - self.init_tcp) * scale

        reward_for_caging = reward_utils.tolerance(
            float(np.linalg.norm(gripper_error)),
            bounds=(0, 0.01),
            margin=np.linalg.norm(gripper_error_init),
            sigmoid="long_tail",
        )

        reward = reward_for_caging + reward_for_opening
        reward *= 5.0

        return (
            reward,
            float(np.linalg.norm(handle - gripper)),
            obs[3],
            handle_error,
            reward_for_caging,
            reward_for_opening,
        )
