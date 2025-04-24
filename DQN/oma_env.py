import numpy as np

class IRSOMAEnv:
    """
    Environment for simulating an IRS-assisted OMA system.

    Assumes a simplified model where the IRS phase shift affects a single
    representative user link (interference=0). The reward is scaled by
    1/num_users to reflect the resource sharing fraction in OMA, making it
    comparable to systems using the full resource block.
    """
    def __init__(self, num_users=10, resolution_bits=5, num_elements=5, signal_power=10):
        """
        Initializes the OMA environment.

        Args:
            num_users (int): Number of users sharing the resource orthogonally.
            resolution_bits (int): Number of bits for IRS phase resolution.
            num_elements (int): Number of reflecting elements on the IRS.
            signal_power (float): Transmit signal power (linear scale) used during
                                 a user's transmission slot/band.
        """
        self.num_users = num_users
        self.B = resolution_bits
        # Ensure signal_power is treated as linear if converted outside
        self.signal_power = signal_power
        self.num_elements = num_elements
        self.phase_levels = 2 ** self.B
        self.state_dim = self.num_users * 2
        self.action_dim = self.phase_levels
        self.time = 0
        self.reset()

    def reset(self):
        """
        Resets the environment state and time step.

        Returns:
            np.ndarray: The flattened initial state array.
        """
        self.state = np.random.randn(self.num_users, 2)
        self.time = 0
        return self.state.flatten()

    def step(self, action):
        """
        Executes one time step within the environment.

        Args:
            action (int): The discrete action chosen by the agent.

        Returns:
            tuple: A tuple containing: (next_state, reward, done).
        """
        phase_shift = 2 * np.pi * action / self.phase_levels
        irs_gain = np.abs(np.cos(phase_shift))
        # Calculate SINR assuming ZERO interference for the active OMA user
        sinr = self._calculate_sinr(irs_gain)

        # Calculate the rate achieved during the user's transmission time
        instantaneous_rate = np.log2(1 + sinr)

        # --- Reward is the average rate contribution, scaled by resource fraction ---
        # If num_users is 0 or 1, avoid division issues / handle appropriately
        if self.num_users > 0:
            reward = instantaneous_rate / self.num_users
            # reward = instantaneous_rate

        else:
            reward = instantaneous_rate # Or handle as an error/special case
        # -------------------------------------------------------------------------

        self.time += 1
        done = self.time > 30

        return self.state.flatten(), reward, done

    def _calculate_sinr(self, gain):
        """
        Calculates the SINR for a single OMA link (interference=0).

        Args:
            gain (float): The effective channel gain provided by the IRS.

        Returns:
            float: The calculated Signal-to-Noise Ratio.
        """
        interference = 0  # OMA has no interference in its own slot/band
        noise = 1         # Normalized noise power
        scaled_gain = gain * (self.num_elements ** 2)
        # Use the linear signal power
        return scaled_gain * self.signal_power / (interference + noise + 1e-10)