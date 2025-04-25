import numpy as np

class IRSNOMAEnv:
    def __init__(self, num_users=10, num_clusters=5, num_elements=25, resolution_bits=5, max_power=10):
        assert num_users % num_clusters == 0, "Users must be divisible by number of clusters"
        
        self.num_users = num_users
        self.num_clusters = num_clusters
        self.users_per_cluster = num_users // num_clusters
        self.num_elements = num_elements
        self.B = resolution_bits
        self.max_power = max_power  # total transmit power budget
        self.phase_levels = 2 ** self.B

        # State: user channel info
        self.state_dim = self.num_users * 2

        # Action: [phase shift (1), power per cluster (num_clusters)]
        self.action_dim = 1 + self.num_clusters
        self.reset()

    def reset(self):
        # Each user's channel: simplified as random vector (e.g., position)
        self.state = np.random.randn(self.num_users, 2)
        self.time = 0
        return self.state.flatten()

    def step(self, action):
        # Extract discrete phase action
        phase_action = int(action[0]) % self.phase_levels

        # Power per cluster (normalized and scaled to max_power)
        cluster_powers = np.clip(action[1:], 0, 1)
        cluster_powers = cluster_powers / np.sum(cluster_powers) * self.max_power

        # Apply IRS phase shift
        phase_shift = 2 * np.pi * phase_action / self.phase_levels
        irs_gain = np.abs(np.cos(phase_shift))  # simplified IRS gain model

        # Calculate SINR per user based on their cluster's power
        sinrs = []
        for i in range(self.num_clusters):
            cluster_users = range(i * self.users_per_cluster, (i + 1) * self.users_per_cluster)
            cluster_power = cluster_powers[i]

            # Simplified: divide power among users (in practice, allocate more to weaker users)
            power_per_user = cluster_power / self.users_per_cluster
            for u in cluster_users:
                sinr = self._calculate_sinr(irs_gain, power_per_user)
                sinrs.append(sinr)

        reward = np.sum(np.log2(1 + np.array(sinrs)))  # Sum-rate
        done = self.time > 30
        self.time += 1
        return self.state.flatten(), reward, done

    def _calculate_sinr(self, gain, power):
        interference = 5
        noise = 1
        scaled_gain = gain * (self.num_elements ** 2)
        return scaled_gain * power / (interference + noise)