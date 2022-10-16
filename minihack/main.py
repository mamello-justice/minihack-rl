#!/bin/bash

import gym
import minihack


def main():
    env = gym.make("MiniHack-River-v0", render_mode="human")
    state = env.reset() # each reset generates a new environment instance
    
    for i in range(10000):
        state_, reward, done, _ = env.step(1)  # move agent '@' north
        env.render()
        
        if done:
            break


if __name__ == "__main__":
    main()
