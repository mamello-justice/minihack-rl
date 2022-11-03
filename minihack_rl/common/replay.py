import numpy as np


class Replay:
    def __init__(self, env, size) -> None:
        self._maxsize = size
        self._mem_counter = 0
        self._state_shape = env.observation_space.shape
        self._num_actions = env.action_space.n

        self.states = np.zeros(
            shape=(self._maxsize, *self._state_shape), dtype=np.float32)
        self.actions = np.zeros(self._maxsize, dtype=np.int64)
        self.rewards = np.zeros(self._maxsize, dtype=np.float32)
        self.next_states = np.zeros(
            shape=(self._maxsize, *self._state_shape), dtype=np.float32)
        self.terminals = np.zeros(self._maxsize, dtype=np.float32)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.terminals[idx]
        )

    def push(self, state, action, reward, next_state, terminal):
        next_idx = self._mem_counter % self._maxsize

        self.states[next_idx] = state
        self.actions[next_idx] = action
        self.rewards[next_idx] = reward
        self.next_states[next_idx] = next_state
        self.terminals[next_idx] = terminal

        self._mem_counter += 1

    def sample(self, N):
        memory = min(self._mem_counter, self._maxsize) - 1
        assert self._mem_counter >= N, f"memory={memory} must have at least N={N} frames"

        indices = np.random.randint(0, memory, size=N)
        return self[indices]
