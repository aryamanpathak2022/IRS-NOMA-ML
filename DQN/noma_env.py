import numpy as np

class IRSNOMAEnv:
    def __init__(self, num_users=10, num_clusters=5, num_elements=25, resolution_bits=5, signal_power=10):
        self.num_users = num_users
        self.num_clusters = num_clusters
        self.num_elements = num_elements
        self.B = resolution_bits
        self.signal_power = signal_power  # store signal power
        self.phase_levels = 2 ** self.B
        self.state_dim = self.num_users * 2  # could be channel/position info
        self.action_dim = self.phase_levels  # discretized phase shift
        self.reset()

    def reset(self):
        # Reset state: simulate channel state as a proxy for user distribution
        self.state = np.random.randn(self.num_users, 2)
        self.time = 0
        return self.state.flatten()

    def step(self, action):
        # Apply discrete phase shift: simulate effect on SINR
        phase_shift = 2 * np.pi * action / self.phase_levels
        irs_gain = np.abs(np.cos(phase_shift))  # simplified effect
        sinr = self._calculate_sinr(irs_gain)
        reward = np.log2(1 + sinr)  # Sum rate
        done = self.time > 30
        self.time += 1
        return self.state.flatten(), reward, done

    def _calculate_sinr(self, gain):
        interference = 5
        noise = 1

        # Scale gain by number of IRS elements squared (ideal scenario)
        scaled_gain = gain * (self.num_elements ** 2)

        return scaled_gain * self.signal_power / (interference + noise)