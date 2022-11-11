# Copyright (c) Facebook, Inc. and its affiliates.

from collections import OrderedDict
from typing import Tuple, Union

import gym
import numpy as np
import threading

from minihack.agent.common.envs import tasks


class RLLibNLEEnv(gym.Env):
    def __init__(self, env_config: dict) -> None:
        # We sort the observation keys so we can create the OrderedDict output
        # in a consistent order
        self._observation_keys = sorted(
            env_config.get("observation_keys", ("blstats", "glyphs"))
        )
        self._flags = env_config["flags"]
        self.gym_env = self.create_env()

    def create_env(self):
        # Create environment instances for actors
        with threading.Lock():
            env_class = tasks.ENVS[self._flags["env"]]

            if self._flags["save_tty"]:
                savedir = ""  # NLE choses location
            else:
                savedir = None

            kwargs = dict(
                savedir=savedir,
                observation_keys=self._flags["observation_keys"],
                actions=self._flags["supported_actions"],
            )
            kwargs.update(self._flags['reward_config'])
            if not tasks.is_env_minihack(env_class):
                kwargs.update(max_episode_steps=self._flags["max_num_steps"])
                kwargs.update(character=self._flags["character"])

            env = env_class(**kwargs)

            return env

    @property
    def action_space(self) -> gym.Space:
        return self.gym_env.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self.gym_env.observation_space

    def reset(self) -> dict:
        return self._process_obs(self.gym_env.reset())

    def _process_obs(self, obs: dict) -> dict:
        return OrderedDict({key: obs[key] for key in self._observation_keys})

    def step(
        self, action: Union[int, np.int64]
    ) -> Tuple[dict, Union[np.number, int], Union[np.bool_, bool], dict]:
        obs, reward, done, info = self.gym_env.step(action)
        return self._process_obs(obs), reward, done, info

    def render(self):
        return self.gym_env.render()

    def close(self):
        return self.gym_env.close()
