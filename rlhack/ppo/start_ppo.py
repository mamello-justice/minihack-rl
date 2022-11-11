import gym
import nle  # noqa: F401
import minihack  # noqa: F401
from nle import nethack
import random
import numpy as np
import pygame
import torch
from VideoRecorderCallback import VideoRecorderCallback, ImageRecorderCallback, FigureRecorderCallback, OnnxablePolicy
import torch as th
# fake rendering
# Set up fake display; otherwise rendering will fail
import os
# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'
## 
import stable_baselines3
print(torch.cuda.is_available())

print(stable_baselines3.__version__)
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.callbacks import BaseCallback

from minihack import RewardManager

MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS = MOVE_ACTIONS + (
    nethack.Command.OPEN,
    nethack.Command.KICK,
    nethack.Command.SEARCH,
)
SOME_ACTIONS = NAVIGATE_ACTIONS + (
    nethack.Command.ZAP,
    nethack.Command.WEAR,
    nethack.MiscDirection.UP,
    nethack.MiscDirection.DOWN,
    nethack.Command.SEARCH,
    nethack.Command.PICKUP,
    nethack.Command.MOVE,
    nethack.Command.INVENTORY,
    nethack.Command.MONSTER,
    nethack.Command.THROW,
    nethack.Command.LOOK,
    nethack.Command.OPEN,

)
# print(NAVIGATE_ACTIONS)
# print(MOVE_ACTIONS)
print(SOME_ACTIONS)

# PPO Minihack environment

# Define a reward manager
reward_manager = RewardManager()


reward_manager.add_wear_event("wear", reward=0)
reward_manager.add_wield_event("dagger", reward=0.1)
reward_manager.add_wield_event("wand", reward=1)
reward_manager.add_kill_event("demon",
    reward=5, terminal_required=False)
reward_manager.add_kill_event("monster",
    reward=5, terminal_required=False)
reward_manager.add_location_event("door",
    reward=1, terminal_required=False)
# reward_manager.add
# TO DO 
# reward_manager.add_custom_reward_fn()


# -1 reward for standing on a sink
# but isn't required for terminating
# the episode
# reward_manager.add_
# reward_manager.add_wod_event("death", reward=-100)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#weights and biases import and dashboard

# obs_crop_h and obs_crop_w to specify crop height and width
env = gym.make(
    "MiniHack-Quest-Hard-v0",
    observation_keys=("pixel_crop", "chars_crop", "colors_crop", "blstats", "message"),
    reward_manager=reward_manager,
    actions=SOME_ACTIONS,
    reward_win = 5,
    reward_lose = -100
)
env = gym.wrappers.RecordEpisodeStatistics(env)
# env = gym.wrappers.RecordVideo(env, "videoss", episode_trigger =lambda t:t%2  == 0)

# put this in seperate class and import it. 
def make_env(gym_id):
    def think():
        env = gym.make()
    return think

# env = gym.make("MiniHack-River-v0")
# env = gym.vector.AsyncVectorEnv('MiniHack-Quest-Medium-v0')
env.reset()
env.step(1)
env.render()

# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MultiInputPolicy", env, 
            learning_rate=0.00001, 
            n_steps=4092, 
            batch_size=32, 
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95, 
            clip_range=0.2, 
            clip_range_vf=None, 
            normalize_advantage=True, 
            ent_coef=0.0001, 
            vf_coef=0.5, 
            max_grad_norm=0.5, 
            use_sde=False, 
            sde_sample_freq=-1,
            target_kl=None, 
            tensorboard_log="logs/",
            policy_kwargs=None,
            verbose=1, 
            seed=0,
            device='cuda', 
            _init_setup_model=True
            )
# video_recorder = VideoRecorderCallback(env, render_freq=500)

# model.learn(total_timesteps=10000, callback = video_recorder)
model.learn(total_timesteps=10000, callback = FigureRecorderCallback())

model.save("ppo_minihack")

# del model # remove to demonstrate saving and loading
print("Done learning")
model = PPO.load("ppo_minihack")

# onnxable_model = OnnxablePolicy(
#     model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net
# )


obs = env.reset()

while True:
    
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # print(obs['message'])
    # print("Observation is ", obs)
    if dones:
        print(f"Episodic return : {info['episode']}")
        env.render()
        break

# evaluate the policy 

mean_reward, std_reward = evaluate_policy(model, Monitor(env), n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# video record

# important actions

# 1. ZAP
# 2. WEAR
# 3. Wield
# 4. UP
# 5. UNTRAP
# 6. TURN
# 7. THROW
# 8. TELEPORT
# 9. SEARCH
# 10. RUB
# 11. PUTON
# 12. PICKUP
# 13. open
# 14. MOVE
# 15. MOVEFAR
# 16. MONSTER
# 17. LOOK 
# 18. JUMP
# 19. KICK
# 20. INVOKE
# 21. ENHANCE
# 22. EAT
# 23. DOWN 
# 24. CLOSE 
# 25. CAST
# 26. AUTOPICKUP
# 27. APPLY
# 28. 

# monitor_file = os.path.join(str(tmp_path), f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
# monitor_env = Monitor(env, monitor_file)

# well tried but strange error 
# ValueError: max() arg is an empty sequence


def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = DummyVecEnv([lambda: gym.make('MiniHack-Quest-Hard-v0')])
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  # Close the video recorder
#   eval_env.close()

  while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = eval_env.step(action)
    print("Observation2 is ", obs)
    if dones:
        print(f"Episodic return2 : {info['episode']}")
        env.render()
        break

# record_video('Hard-v0', model, video_length=500, prefix='ppo-minihack')