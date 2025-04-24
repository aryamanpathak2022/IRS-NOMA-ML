import numpy as np

class IRSOMAEnv:
    def __init__(self, num_users=10, resolution_bits=5,num_elements=5, signal_power=10):
        self.num_users = num_users
        self.B = resolution_bits
        self.signal_power = signal_power  # store signal power
        self.num_elements = num_elements
        self.phase_levels = 2 ** self.B
        self.state_dim = self.num_users * 2
        self.action_dim = self.phase_levels
        self.reset()

    def reset(self):
        self.state = np.random.randn(self.num_users, 2)
        self.time = 0
        return self.state.flatten()

    def step(self, action):
        phase_shift = 2 * np.pi * action / self.phase_levels
        irs_gain = np.abs(np.cos(phase_shift))
        sinr = self._calculate_sinr(irs_gain)
        reward = np.sum(np.log2(1 + sinr)) / self.num_users
        done = self.time > 30
        self.time += 1
        return self.state.flatten(), reward, done

    def _calculate_sinr(self, gain):
        interference = 0  # ideally zero in OMA
        noise = 1
        scaled_gain = gain * (self.num_elements ** 2)
        return scaled_gain * self.signal_power / (interference + noise)