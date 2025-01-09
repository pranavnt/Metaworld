import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, Union
from gymnasium.core import ObsType

class FilterDeadFeaturesWrapper(gym.Wrapper):
    """Wrapper that removes dead features from the observation space."""

    def __init__(self, env):
        super().__init__(env)
        # Use the environment's alive indices
        self.alive_indices = self.env.alive_indices

        # Keep original observation space for when shrink=False
        self.original_observation_space = env.observation_space

        # Update wrapped observation space for when shrink=True (default)
        self.observation_space = gym.spaces.Box(
            low=self.original_observation_space.low[self.alive_indices],
            high=self.original_observation_space.high[self.alive_indices],
            dtype=self.original_observation_space.dtype
        )

    def step(self, action: np.ndarray, shrink: bool = True) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Step environment and optionally filter observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['full_obs'] = obs.copy()

        if shrink:
            obs = self.shrink_obs(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, shrink: bool = True, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset environment and optionally filter observation."""
        obs, info = self.env.reset(**kwargs)
        info['full_obs'] = obs.copy()

        if shrink:
            obs = self.shrink_obs(obs)

        return obs, info

    @property
    def full_observation_space(self):
        """Access to the original observation space."""
        return self.original_observation_space

    def shrink_obs(self, obs: np.ndarray) -> np.ndarray:
        """Pure function to remove dead features from observation."""
        if obs.ndim == 1:
            return obs[self.alive_indices]
        return obs[:, self.alive_indices]

    def unshrink_obs(self, filtered_obs: np.ndarray) -> np.ndarray:
        """Pure function to reconstruct full observation from filtered one."""
        if filtered_obs.ndim == 1:
            full_obs = np.zeros(self.original_observation_space.shape[0])
            full_obs[self.alive_indices] = filtered_obs
        else:
            full_obs = np.zeros((filtered_obs.shape[0],) + self.original_observation_space.shape)
            full_obs[:, self.alive_indices] = filtered_obs
        return full_obs

class FilteredExpertWrapper:
    """Wrapper for expert policy that handles both filtered and unfiltered observations."""

    def __init__(self, expert_policy, alive_indices):
        self.expert_policy = expert_policy
        self.alive_indices = alive_indices
        self.full_obs_dim = 39  # Based on MetaWorld drawer env

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get expert action, handling both filtered and full observations."""
        # Check if observation is filtered by comparing dimensions
        if len(obs) == len(self.alive_indices):
            full_obs = np.zeros(self.full_obs_dim)
            full_obs[self.alive_indices] = obs
        else:
            full_obs = obs

        return self.expert_policy.get_action(full_obs)
