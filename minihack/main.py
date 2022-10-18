#!/bin/python

import argparse
import gym
import minihack
from environments.human import HumanEnv


def main():
    parser = argparse.ArgumentParser(description=f'MiniHack')
    parser.add_argument('--human',
                        action='store_true',
                        help='Render in human mode')

    args = vars(parser.parse_args())

    env = gym.make("MiniHack-Quest-Hard-v0",
                   observation_keys=("pixel", "message"))

    if args['human']:
        env = HumanEnv(env)

    state = env.reset()
    env.render(False)

    for i in range(10000):
        action = env.action_space.sample()
        state_, reward, done, _ = env.step(action)

        if done:
            break


if __name__ == "__main__":
    main()
