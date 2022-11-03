import argparse
import gym
import minihack

from environments.human import HumanEnv

from dqn.train import train as train_dqn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'MiniHack')
    parser.add_argument('--human',
                        action='store_true',
                        help='Render in human mode')
    parser.add_argument('--n-replay', )

    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument('--dqn', action='store_true')

    args = vars(parser.parse_args())

    env = gym.make("MiniHack-Quest-Hard-v0",
                   observation_keys=("pixel", "message"))

    if args['human']:
        env = HumanEnv(env)

    if args['human']:
        state = env.reset()
        env.render(False)

    if args['dqn']:
        train_dqn(env, args)
