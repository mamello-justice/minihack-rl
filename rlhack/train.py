# Copyright (c) Facebook, Inc. and its affiliates.

import hydra
import logging

import gym
import torch

from nle.env import base

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.air.callbacks import wandb
from ray.tune.registry import register_env

from omegaconf import DictConfig, OmegaConf

from .algorithms import ALGORITHMS
from .config import get_config
from .models import RLLibNLENetwork  # noqa: F401
from .envs import RLLibNLEEnv  # noqa: F401

base.logger.setLevel(logging.WARN)


class TerminationReporter(tune.CLIReporter):
    """Reports only on experiment termination."""

    def should_report(self, _, done=False):
        return done


@hydra.main(config_path=".", config_name="config", version_base='1.2')
def train(cfg: DictConfig) -> None:
    ray.init(num_gpus=cfg.num_gpus, num_cpus=cfg.num_cpus + 1)

    # Register custom environment
    register_env("RLlibNLE-v0", RLLibNLEEnv)

    # Register custom model
    ModelCatalog.register_custom_model("rllib_nle_model", RLLibNLENetwork)

    try:
        Algo = ALGORITHMS[cfg.algo]
    except KeyError as error:
        raise ValueError(
            "The algorithm you specified isn't currently supported: %s",
            cfg.algo,
        ) from error

    config = get_config(cfg, Algo)
    callbacks = []
    if cfg.wandb.active:
        callbacks.append(
            wandb.WandbLoggerCallback(
                project=cfg.wandb.project,
                api_key_file=cfg.wandb.api_key_file,
                entity=cfg.wandb.entity,
                group=cfg.env,
                tags=cfg.wandb.tags,
                save_checkpoints=cfg.wandb.save_checkpoints,
            )
        )

    tune.run(
        Algo,
        stop={"timesteps_total": cfg.total_steps},
        config=config,
        callbacks=callbacks,
        progress_reporter=TerminationReporter(),
        log_to_file=True,
        reuse_actors=True
    )


if __name__ == "__main__":
    train()
