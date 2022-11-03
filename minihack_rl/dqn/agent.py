import random
import torch
import torch.nn as nn
import numpy as np

from tqdm.auto import trange

from minihack_rl.common.replay import Replay
from minihack_rl.dqn.duel_dqn import DuelDQN


class D3QNAgent:
    def __init__(
            self,
            env,
            N_replay,
            N_batch,
            N_,
            double_dqn=False,
            gamma=1,
            lr=0.001,
            lf=1):
        self.N_replay = N_replay
        self.N_batch = N_batch
        self.N_ = N_
        self.lf = lf
        self.double_dqn = double_dqn
        self.gamma = gamma

        self.replay = Replay(env, self.N_replay)

        self.Q = DuelDQN()
        self.Q_ = DuelDQN()
        self.update_target()    # Copy parameters of Q

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.MSE_loss = nn.MSELoss()

        self.state = None
        self.episodic_rewards = [0.0]

    def update_target(self):
        for Q_, Q in zip(self.Q_.parameters(), self.Q.parameters()):
            Q_._copy(Q)

    def td_update(self):
        states, actions, rewards, next_states, terminals = \
            self.replay.sample(self.N_batch)    # Sample minibatch

        # Not used in gradient calculation
        with torch.no_grad():
            if self.use_double_dqn:
                # Max action using online Q
                target_actions = self.Q.forward(next_states)\
                    .argmax(dim=1, keepdim=True)

                # Get Q value for next state choosing online max action
                target_values = self.Q_.forward(next_states)\
                    .gather(dim=1, index=target_actions)\
                    .flatten()

            else:
                # Get max Q value for next state
                target_values = self.Q_.forward(next_states)\
                    .max(dim=1, keepdim=True)[0].flatten()

            y = rewards + \
                self.gamma * target_values * (1 - terminals)

        # Q value of current state
        values = self.Q.forward(states)\
            .gather(dim=1, index=actions.unsqueeze(-1))\
            .flatten()

        # Loss
        loss = self.loss(values, y).to(self.device)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.idx += 1

        if self.idx % self.N_:
            self.update_target()

        return {'loss': loss.item()}

    def random_action(self):
        return self.env.action_space.sample()

    def greedy_action(self, state):
        state = torch.tensor(
            np.array([state]), dtype=torch.float32, device=self.device) / 255

        with torch.no_grad():
            return self.Q(state).argmax(dim=1, keepdim=True).item()

    def remember(self, state, action, reward, next_state, terminal):
        self.replay.push(state, action, reward, next_state, terminal)

        # Rewards
        self.episodic_rewards[-1] += reward

        if terminal:
            self.episodic_rewards.append(0.0)

    def step(self, action):
        # take step in env
        next_state, reward, terminal, _, _ = self.env.step(action)

        # add state, action, reward, next_state, float(terminal) to reply memory - cast terminal to float
        self.remember(self.state, action, reward, next_state, float(terminal))

        self.state = next_state
        return next_state, reward, terminal

    def train(self, steps, epsilon):
        for t in trange(steps):
            if (random.random() <= epsilon):
                action = self.random_action()
            else:
                action = self.greedy_action(self.state)

            next_state, _, terminal = self.step(action)

            if terminal:
                self.state = self.env.reset()
            else:
                self.state = next_state

            if t % self.lf == 0:
                info = self.td_update()
                print(info)
