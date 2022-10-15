#!/bin/bash

import gym
import minihack


def main():
    env = gym.make("MiniHack-River-v0")
    env.reset() # each reset generates a new environment instance
    env.step(1)  # move agent '@' north
    env.render()


if __name__ == "__main__":
    main()
