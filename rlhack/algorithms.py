
from ray.rllib.algorithms import a2c, dqn, impala, ppo

ALGORITHMS: dict[str, impala.Impala | a2c.A2C | dqn.DQN | ppo.PPO] = {
    "impala": impala.Impala,
    "a2c": a2c.A2C,
    "dqn": dqn.DQN,
    "ppo": ppo.PPO,
}
