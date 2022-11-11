from omegaconf import DictConfig, OmegaConf

from nle.nethack import Command, CompassDirection

from minihack.agent.common.envs import tasks
from minihack.agent import is_env_registered, get_env_shortcut

from ray.tune.utils import merge_dicts
from ray.rllib.algorithms import Algorithm
from ray.rllib.models.catalog import MODEL_DEFAULTS

from .algorithms import ALGORITHMS


MOVE_ACTIONS = tuple(CompassDirection)
NAVIGATE_ACTIONS = MOVE_ACTIONS + (
    Command.APPLY,
    Command.KICK,
    Command.OPEN,
    Command.PICKUP,
    Command.SEARCH,
    Command.TWOWEAPON,
    Command.WEAR,
    Command.WIELD,
    Command.ZAP
)


def pick(object: dict, path: list[str]):
    return {k: v for k, v in object.items() if k in path}


def get_config(cfg, Algo: Algorithm):
    args_config = OmegaConf.to_container(cfg)

    # Algo-specific config. Requires hydra config keys to match rllib exactly
    algo_config = args_config.pop(cfg.algo)

    # Remove unnecessary config keys
    for algo in ALGORITHMS.keys():
        if algo != cfg.algo:
            args_config.pop(algo, None)

    # Merge config from hydra (will have some rogue keys but that's ok)
    config = args_config    # No longer need defaults

    # check the name of the environment
    if cfg.env not in tasks.ENVS:
        if is_env_registered(cfg.env):
            cfg.env = get_env_shortcut(cfg.env)
        else:
            raise KeyError(
                f"Could not find an environement with a name: {cfg.env}."
            )

    # Update configuration with parsed arguments in specific ways
    config = merge_dicts(
        config,
        {
            "env": "RLlibNLE-v0",
            "env_config": {
                "flags": merge_dicts(OmegaConf.to_object(cfg),
                                     {"supported_actions": NAVIGATE_ACTIONS}
                                     ),
                "observation_keys": cfg.observation_keys,
                "name": cfg.env,
            },
            "model": merge_dicts(
                MODEL_DEFAULTS,
                {
                    "custom_model": "rllib_nle_model",
                    "custom_model_config": {
                        "flags": cfg,
                        "algo": cfg.algo
                    },
                    "use_lstm": cfg.use_lstm,
                    "lstm_use_prev_reward": True,
                    "lstm_use_prev_action": True,
                    "lstm_cell_size": cfg.hidden_dim,
                },
            ),
            "num_workers": cfg.num_cpus,
            "num_envs_per_worker": int(cfg.num_actors / cfg.num_cpus),
            "rollout_fragment_length": cfg.unroll_length,
        },
    )

    # Merge algo-specific config at top level
    config = merge_dicts(config, algo_config)

    # Pick known keys
    known_keys = Algo.get_default_config().keys()
    config = pick(config, known_keys)

    return OmegaConf.to_object(OmegaConf.create(config))
