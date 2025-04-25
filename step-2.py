from LSTM.lstm_predictor import *      
from CLUSTERING.kgmm_clustering import *
from DQN.agent_2 import *
from DQN.noma_env_2 import *
import time

import matplotlib.pyplot as plt

# Step 1: Simulate
print("Simulating user trajectories...")
trajectories = generate_user_trajectories(num_users=10, steps=50)
print("User trajectories generated. Shape:", trajectories.shape)

# Step 2: Train LSTM
print("Training LSTM model...")
lstm_model = train_lstm_model(trajectories, seq_len=10)
print("LSTM model trained.")

# Step 3: Predict future positions
print("Predicting future positions using LSTM...")
predicted = predict_positions(lstm_model, trajectories, future_steps=10)
print("Prediction completed. Shape of predicted positions:", predicted.shape)

# Step 4: Visualize
print("Plotting actual vs predicted trajectories...")
plot_trajectories(trajectories[:, -10:, :], predicted)

# Use final predicted positions for clustering (shape: num_users x 2)
last_pred_positions = predicted[:, -1, :]
print("Last predicted positions extracted. Shape:", last_pred_positions.shape)

# Perform clustering
print("Performing K-GMM clustering...")
labels, centers = k_gmm_clustering(last_pred_positions, n_clusters=5)
print("Clustering completed. Labels:", labels)

# Visualize clusters
print("Plotting clusters...")
plot_clusters(last_pred_positions, labels, centers)

import numpy as np
import matplotlib.pyplot as plt
import time
import torch # Import torch if agent uses it

# --- Assumed Imports ---
# Make sure the updated DQNAgent class and IRSNOMAEnv class are defined
# in separate files and can be imported correctly.
# Example:
try:
    # Make sure these filenames match your actual files
    from DQN.agent_2 import *
    from DQN.noma_env_2 import *
except ImportError as e:
    print(f"Error importing environment/agent classes: {e}")
    print("Please ensure irs_noma_env.py and dqn_optimizer.py are in the Python path")
    # Add fallback dummy classes if needed for testing script structure
    class IRSNOMAEnv:
        def __init__(self, num_elements=10, max_power=0.1, num_users=10, num_clusters=5, resolution_bits=5): # Matched user's env init
            self.state_dim = num_users * 2
            self.num_clusters = num_clusters
            self.phase_levels = 2**resolution_bits
            self.num_elements = num_elements
            self.max_power = max_power
            self.noise = 1 # Match user env
            self.interference = 5 # Match user env
            print(f"Dummy Env Init: Elements={num_elements}, Power={max_power}")
        def reset(self): return np.zeros(self.state_dim)
        def step(self, action): return np.zeros(self.state_dim), 0.1 * self.num_elements * self.max_power, False # Dummy reward
    class DQNAgent:
         def __init__(self, state_dim, phase_levels, num_clusters, total_power, # No batch_size
                      lr=1e-3, gamma=0.99, epsilon=1.0, decay=0.995,
                      epsilon_min=0.01, target_update_freq=100, buffer_size=10000):
            self.state_dim = state_dim
            print(f"Dummy Agent Init: StateDim={state_dim}, Power={total_power}")
         def select_action(self, state): return np.random.rand(1 + 5) # Dummy action shape
         def store(self, s, a, r, ns, d): pass
         def train(self, batch_size=32): return 0.5 # Dummy loss
# -----------------------

# --- Simulation Parameters ---
# Using dBm for definition, converting to Watts for env/agent
powers_dbm = [ 15, 20, 25, 30,35,40]
powers_watts = [10**((p - 30) / 10) for p in powers_dbm]

elements = [5, 10, 15, 20, 25, 30]
num_runs = 10  # Number of independent runs to average over
fixed_iterations = 300 # Fixed number of training steps for plots 2 and 3
TRAINING_BATCH_SIZE = 32 # Define batch size for agent.train() separately

# --- Fixed Environment Parameters (Match your Env class defaults/needs) ---
env_defaults = {
    'num_users': 10,
    'num_clusters': 5,
    'resolution_bits': 5, # Matched your Env definition
    # Add any other relevant fixed parameters your env might need
}
# Define noise power explicitly based on your Env's _calculate_sinr
NOISE_OMA = 1.0 # From your env's _calculate_sinr
INTERFERENCE_OMA = 0.0 # By definition for OMA baseline

# --- Agent Hyperparameters (Consistent Set - NO BATCH SIZE HERE) ---
agent_params = {
    'lr': 1e-4,  # Using the value from previous script
    'gamma': 0.99,
    'epsilon': 1.0,         # Initial epsilon
    'decay': 0.995,         # Epsilon decay rate per step
    'epsilon_min': 0.01,
    'target_update_freq': 100,
    'buffer_size': 10000,
}


# ==============================================================================
# 2. Sum Rate vs IRS Elements with Different Powers (Averaged)
# ==============================================================================
print("\nStarting Simulation 2: Sum Rate vs IRS Elements (Averaged)...")
start_time_2 = time.time()

# Structure: {power_dbm: [ [run1_elem1, run1_elem2,...], [run2_elem1, run2_elem2,...], ... ]}
all_runs_avg_sum_rates_elements = {p_dbm: [] for p_dbm in powers_dbm}

for run in range(num_runs):
    print(f"\n--- Run {run + 1} / {num_runs} --- (Plot 2)")
    # Structure: {power_dbm: [elem1_rate, elem2_rate, ...]}
    run_avg_sum_rates_elements = {p_dbm: [] for p_dbm in powers_dbm}

    for p_dbm, p_watts in zip(powers_dbm, powers_watts):
        print(f"  Power level: {p_dbm} dBm ({p_watts:.4f} W)")
        element_rates_this_power = [] # Temp list for rates at this power level

        for n_elements in elements:
            # print(f"    Number of IRS elements: {n_elements}") # Verbose

            # --- Initialize Environment and Agent ---
            try:
                # Pass max_power (Watts) during environment initialization
                env = IRSNOMAEnv(num_elements=n_elements,
                                 max_power=p_watts, # Use Watts here
                                 **env_defaults)

                # Pass total_power (Watts) during agent initialization
                agent = DQNAgent(state_dim=env.state_dim,
                                 phase_levels=env.phase_levels,
                                 num_clusters=env.num_clusters,
                                 total_power=p_watts, # Use Watts here
                                 **agent_params) # Pass other hyperparameters (NO batch_size)

            except Exception as e:
                print(f"ERROR initializing env/agent: {e}")
                print(f"Skipping combination: Power={p_dbm}dBm, Elements={n_elements}")
                element_rates_this_power.append(np.nan) # Mark as invalid
                continue # Skip to next element count

            # --- Training Loop ---
            state = env.reset()
            total_reward = 0
            steps_in_run = 0

            for i in range(fixed_iterations): # Use fixed iterations
                action = agent.select_action(state)
                try:
                    next_state, reward, done = env.step(action)
                except Exception as e:
                     print(f"ERROR during env.step: {e}")
                     total_reward = np.nan # Mark run as invalid
                     steps_in_run = 1 # Avoid division by zero
                     break

                agent.store(state, action, reward, next_state, done)
                agent.train(batch_size=TRAINING_BATCH_SIZE) # Pass batch size HERE
                state = next_state
                total_reward += reward
                steps_in_run += 1
                if done:
                    # print(f"    Episode ended early at step {i}, resetting.") # Verbose
                    state = env.reset()

            # Calculate average reward for this specific (element, power) setting in this run
            if steps_in_run > 0 and not np.isnan(total_reward):
                avg_reward_this_run = total_reward / steps_in_run
            else:
                avg_reward_this_run = np.nan # Invalid result

            element_rates_this_power.append(avg_reward_this_run)
            # print(f"      Elements={n_elements}, Avg Rate={avg_reward_this_run:.4f}") # Verbose

        # Store the list of rates for this power level for the current run
        run_avg_sum_rates_elements[p_dbm] = element_rates_this_power

    # Append the results of the entire run to the master list
    for p_dbm in powers_dbm:
        all_runs_avg_sum_rates_elements[p_dbm].append(run_avg_sum_rates_elements[p_dbm])

# --- Averaging Results Across Runs (Plot 2) ---
print("\n--- Averaging Results Across Runs (Plot 2) ---")
final_avg_elements = {p_dbm: [] for p_dbm in powers_dbm}
final_std_elements = {p_dbm: [] for p_dbm in powers_dbm} # Optional: Standard deviation

for p_dbm in powers_dbm:
    # Shape: (num_runs, num_elements_points)
    rates_array = np.array(all_runs_avg_sum_rates_elements[p_dbm], dtype=float)
    # Use nanmean/nanstd to ignore NaN values from potential errors
    final_avg_elements[p_dbm] = np.nanmean(rates_array, axis=0).tolist()
    final_std_elements[p_dbm] = np.nanstd(rates_array, axis=0).tolist()

# --- Plotting Averaged Results (Plot 2) ---
print("\nPlotting Averaged Results (Plot 2: Rate vs Elements)...")
plt.figure(figsize=(10, 7))
markers = ['o', 's', '^', 'd', 'v', '*']
colors_plot2 = plt.cm.plasma(np.linspace(0, 1, len(powers_dbm)))

# Check if any valid data exists before plotting
has_valid_data_p2 = not all(all(np.isnan(val) for val in final_avg_elements[p_dbm]) for p_dbm in powers_dbm)

if has_valid_data_p2:
    for i, p_dbm in enumerate(powers_dbm):
        if not all(np.isnan(final_avg_elements[p_dbm])): # Check if data for this power level exists
            plt.plot(elements, final_avg_elements[p_dbm],
                     label=f"Power {p_dbm} dBm", marker=markers[i % len(markers)],
                     color=colors_plot2[i], linewidth=2, markersize=6)
            # Optional: Add shaded region for standard deviation
            lower_bound = np.array(final_avg_elements[p_dbm]) - np.array(final_std_elements[p_dbm])
            upper_bound = np.array(final_avg_elements[p_dbm]) + np.array(final_std_elements[p_dbm])
            plt.fill_between(elements, lower_bound, upper_bound, alpha=0.15, color=colors_plot2[i])

    plt.xlabel("Number of IRS Elements", fontsize=12)
    plt.ylabel("Average Sum Rate (bps/Hz)", fontsize=12)
    plt.title(f"Avg Sum Rate vs IRS Elements (Averaged over {num_runs} runs, {fixed_iterations} iter.)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--')
    plt.xticks(elements)
    plt.yticks(fontsize=10)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.tight_layout()
    plt.show()
else:
    print("Skipping Plot 2: No valid data generated.")


end_time_2 = time.time()
print(f"\nSimulation 2 time: {end_time_2 - start_time_2:.2f} seconds")


import numpy as np
import matplotlib.pyplot as plt
import time
import torch # Import torch if agent uses it

# --- Assumed Imports ---
# Make sure the updated DQNAgent class and IRSNOMAEnv class are defined
# in separate files and can be imported correctly.
# Example:
try:
    # Make sure these filenames match your actual files
    from DQN.agent_2 import *
    from DQN.noma_env_2 import *
except ImportError as e:
    print(f"Error importing environment/agent classes: {e}")
    print("Please ensure irs_noma_env.py and dqn_optimizer.py are in the Python path")
    # Add fallback dummy classes if needed for testing script structure
    class IRSNOMAEnv:
        def __init__(self, num_elements=10, max_power=0.1, num_users=10, num_clusters=5, resolution_bits=5): # Matched user's env init
            self.state_dim = num_users * 2
            self.num_clusters = num_clusters
            self.phase_levels = 2**resolution_bits
            self.num_elements = num_elements
            self.max_power = max_power
            self.noise = 1 # Match user env
            self.interference = 5 # Match user env
            print(f"Dummy Env Init: Elements={num_elements}, Power={max_power}")
        def reset(self): return np.zeros(self.state_dim)
        def step(self, action): return np.zeros(self.state_dim), 0.1 * self.num_elements * self.max_power, False # Dummy reward
    class DQNAgent:
         def __init__(self, state_dim, phase_levels, num_clusters, total_power, # No batch_size
                      lr=1e-3, gamma=0.99, epsilon=1.0, decay=0.995,
                      epsilon_min=0.01, target_update_freq=100, buffer_size=10000):
            self.state_dim = state_dim
            print(f"Dummy Agent Init: StateDim={state_dim}, Power={total_power}")
         def select_action(self, state): return np.random.rand(1 + 5) # Dummy action shape
         def store(self, s, a, r, ns, d): pass
         def train(self, batch_size=32): return 0.5 # Dummy loss
# -----------------------

# --- Simulation Parameters ---
# Using dBm for definition, converting to Watts for env/agent
powers_dbm = [5, 10, 15, 20, 25, 30]
powers_watts = [10**((p - 30) / 10) for p in powers_dbm]

elements = [5, 10, 15, 20, 25, 30]
num_runs = 10  # Number of independent runs to average over
fixed_iterations = 300 # Fixed number of training steps for plots 2 and 3
TRAINING_BATCH_SIZE = 32 # Define batch size for agent.train() separately

# --- Fixed Environment Parameters (Match your Env class defaults/needs) ---
env_defaults = {
    'num_users': 10,
    'num_clusters': 5,
    'resolution_bits': 5, # Matched your Env definition
    # Add any other relevant fixed parameters your env might need
}
# Define noise power explicitly based on your Env's _calculate_sinr
NOISE_OMA = 1.0 # From your env's _calculate_sinr
INTERFERENCE_OMA = 0.0 # By definition for OMA baseline

# --- Agent Hyperparameters (Consistent Set - NO BATCH SIZE HERE) ---
agent_params = {
    'lr': 1e-4,  # Using the value from previous script
    'gamma': 0.99,
    'epsilon': 1.0,         # Initial epsilon
    'decay': 0.995,         # Epsilon decay rate per step
    'epsilon_min': 0.01,
    'target_update_freq': 100,
    'buffer_size': 10000,
}

# ==============================================================================
# 3. Sum Rate vs IRS Elements: NOMA (RL) vs OMA (Analytical) (Averaged)
# ==============================================================================

print("\nStarting Simulation 3: NOMA (RL) vs OMA (Analytical) Comparison (Averaged)...")
start_time_3 = time.time()

# Data structures for NOMA (from RL) and OMA (analytical) results
# Structure: {power_dbm: [ [run1_elem1,...], [run2_elem1,...], ... ]}
all_runs_noma_rates = {p_dbm: [] for p_dbm in powers_dbm}
all_runs_oma_rates = {p_dbm: [] for p_dbm in powers_dbm} # Stores analytical results per run (will be identical)

for run in range(num_runs):
    print(f"\n--- Run {run + 1} / {num_runs} --- (Plot 3)")
    run_noma_rates = {p_dbm: [] for p_dbm in powers_dbm}
    run_oma_rates = {p_dbm: [] for p_dbm in powers_dbm} # Store analytical OMA per run

    for p_dbm, p_watts in zip(powers_dbm, powers_watts):
        print(f"  Power level: {p_dbm} dBm ({p_watts:.4f} W)")

        element_rates_noma_this_power = []
        element_rates_oma_this_power = [] # Store analytical OMA results

        for n_elements in elements:
            # --- NOMA Simulation (RL based) ---
            # print(f"    NOMA (RL): Elements={n_elements}") # Verbose
            noma_init_success = False
            try:
                # Pass max_power (Watts) during environment initialization
                env_noma = IRSNOMAEnv(num_elements=n_elements,
                                     max_power=p_watts, # Use Watts here
                                     **env_defaults)
                # Pass total_power (Watts) during agent initialization
                agent_noma = DQNAgent(state_dim=env_noma.state_dim,
                                     phase_levels=env_noma.phase_levels,
                                     num_clusters=env_noma.num_clusters,
                                     total_power=p_watts, # Use Watts here
                                     **agent_params) # Pass other hyperparameters (NO batch_size)
                noma_init_success = True

            except Exception as e:
                print(f"ERROR initializing NOMA env/agent: {e}")
                print(f"Skipping NOMA combination: Power={p_dbm}dBm, Elements={n_elements}")
                element_rates_noma_this_power.append(np.nan) # Mark as invalid

            # --- NOMA Training Loop ---
            if noma_init_success:
                state = env_noma.reset()
                total_reward_noma = 0
                steps_noma = 0
                for i in range(fixed_iterations):
                    action = agent_noma.select_action(state)
                    try:
                        next_state, reward, done = env_noma.step(action)
                    except Exception as e:
                         print(f"ERROR during NOMA env.step: {e}")
                         total_reward_noma = np.nan
                         steps_noma = 1
                         break

                    agent_noma.store(state, action, reward, next_state, done)
                    agent_noma.train(batch_size=TRAINING_BATCH_SIZE) # Pass batch size HERE
                    state = next_state
                    total_reward_noma += reward
                    steps_noma += 1
                    if done:
                        state = env_noma.reset()

                # Calculate NOMA average rate for this setting
                if steps_noma > 0 and not np.isnan(total_reward_noma):
                     avg_noma = total_reward_noma / steps_noma
                else:
                     avg_noma = np.nan
                element_rates_noma_this_power.append(avg_noma)
            # --- End NOMA Simulation ---


            # --- OMA Analytical Calculation ---
            # print(f"    OMA (Analytical): Elements={n_elements}") # Verbose
            try:
                # Assumptions for Analytical OMA Baseline:
                # 1. Uses gain scaling from your specific IRSNOMAEnv's _calculate_sinr
                #    scaled_gain = gain * (n_elements ** 2)
                # 2. Assumes perfect phase alignment (gain = 1.0) for max potential gain
                # 3. Interference-free (by OMA definition)
                # 4. Uses the *same* noise power as defined in your Env's _calculate_sinr

                max_gain_factor = 1.0 # From np.abs(np.cos(phase_shift)) max value
                scaled_gain_oma_analytical = max_gain_factor * (n_elements ** 2) # Match NOMA env gain scaling

                # Calculate the interference-free SNR for a single representative OMA user
                # Using NOISE_OMA=1 and INTERFERENCE_OMA=0 based on your env
                # Add small epsilon to denominator to prevent division by zero if noise is ever zero
                snr_oma_analytical = (scaled_gain_oma_analytical * p_watts) / (INTERFERENCE_OMA + NOISE_OMA + 1e-20)

                # Calculate Analytical OMA Rate = log2(1 + SNR_single_user)
                # Clamp SNR to avoid potential overflow issues with very high gain/power
                max_snr = 1e12 # Set a reasonable upper limit for SNR
                snr_oma_analytical = min(snr_oma_analytical, max_snr)

                avg_oma_analytical = np.log2(1 + snr_oma_analytical)
                element_rates_oma_this_power.append(avg_oma_analytical)

            except Exception as e:
                print(f"ERROR during OMA calculation: {e}")
                element_rates_oma_this_power.append(np.nan)
            # --- End OMA Analytical Calculation ---

        # Store results for this power level
        run_noma_rates[p_dbm] = element_rates_noma_this_power
        run_oma_rates[p_dbm] = element_rates_oma_this_power

    # Append results for the whole run
    for p_dbm in powers_dbm:
        all_runs_noma_rates[p_dbm].append(run_noma_rates[p_dbm])
        all_runs_oma_rates[p_dbm].append(run_oma_rates[p_dbm])

# --- Averaging Results Across Runs (Plot 3)---
# Averaging NOMA (RL results) smooths RL variance.
# Averaging OMA (analytical results) will just return the deterministic analytical value.
print("\n--- Averaging Results Across Runs (Plot 3) ---")
final_avg_noma = {p_dbm: [] for p_dbm in powers_dbm}
final_std_noma = {p_dbm: [] for p_dbm in powers_dbm}
final_avg_oma = {p_dbm: [] for p_dbm in powers_dbm}
final_std_oma = {p_dbm: [] for p_dbm in powers_dbm} # Will be ~zero for analytical OMA

for p_dbm in powers_dbm:
    noma_rates_array = np.array(all_runs_noma_rates[p_dbm], dtype=float)
    final_avg_noma[p_dbm] = np.nanmean(noma_rates_array, axis=0).tolist()
    final_std_noma[p_dbm] = np.nanstd(noma_rates_array, axis=0).tolist()

    oma_rates_array = np.array(all_runs_oma_rates[p_dbm], dtype=float)
    final_avg_oma[p_dbm] = np.nanmean(oma_rates_array, axis=0).tolist()
    final_std_oma[p_dbm] = np.nanstd(oma_rates_array, axis=0).tolist() # Should be near zero


# --- Plotting Averaged Results (Plot 3a - NOMA RL Only) ---
print("\nPlotting Averaged Results (Plot 3a: NOMA (RL))...")
plt.figure("NOMA Results", figsize=(10, 7)) # Give figure a name
markers = ['o', 's', '^', 'd', 'v', '*']
linestyles = ['-', '--', ':', '-.']
colors_plot3_noma = plt.cm.viridis(np.linspace(0, 1, len(powers_dbm)))

# Check if any valid NOMA data exists
has_valid_noma_p3 = not all(all(np.isnan(val) for val in final_avg_noma[p_dbm]) for p_dbm in powers_dbm)

if has_valid_noma_p3:
    for i, p_dbm in enumerate(powers_dbm):
        color = colors_plot3_noma[i]
        marker = markers[i % len(markers)]

        if not all(np.isnan(final_avg_noma[p_dbm])):
            plt.plot(elements, final_avg_noma[p_dbm],
                     label=f"NOMA (RL) P={p_dbm}dBm", marker=marker,
                     linestyle=linestyles[0], color=color, linewidth=2, markersize=6) # Solid line
            lower_noma = np.array(final_avg_noma[p_dbm]) - np.array(final_std_noma[p_dbm])
            upper_noma = np.array(final_avg_noma[p_dbm]) + np.array(final_std_noma[p_dbm])
            plt.fill_between(elements, lower_noma, upper_noma, alpha=0.15, color=color)

    plt.xlabel("Number of IRS Elements", fontsize=12)
    plt.ylabel("Average Sum Rate (bps/Hz)", fontsize=12)
    plt.title(f"Avg Sum Rate vs IRS Elements: NOMA (RL) (Avg over {num_runs} runs)", fontsize=14)
    plt.legend(fontsize=10, loc='upper left') # Adjusted legend
    plt.grid(True, linestyle='--')
    plt.xticks(elements)
    plt.yticks(fontsize=10)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.tight_layout()
    # plt.show() # Optionally show this plot immediately

else:
    print("Skipping Plot 3a (NOMA): No valid NOMA data generated.")


# --- Plotting Averaged Results (Plot 3b - OMA Analytical Only) ---
print("\nPlotting Averaged Results (Plot 3b: OMA (Analytical))...")
plt.figure("OMA Results", figsize=(10, 7)) # Give figure a different name
# Reuse markers or colors if desired
colors_plot3_oma = plt.cm.plasma(np.linspace(0, 1, len(powers_dbm))) # Different colormap

# Check if any valid OMA data exists
has_valid_oma_p3 = not all(all(np.isnan(val) for val in final_avg_oma[p_dbm]) for p_dbm in powers_dbm)

if has_valid_oma_p3:
    for i, p_dbm in enumerate(powers_dbm):
        color = colors_plot3_oma[i]
        marker = markers[i % len(markers)]

        if not all(np.isnan(final_avg_oma[p_dbm])):
             plt.plot(elements, final_avg_oma[p_dbm],
                      label=f"OMA (Analytical) P={p_dbm}dBm", marker=marker,
                      linestyle=linestyles[0], color=color, linewidth=2, markersize=6) # Solid line
             # No shaded region for analytical OMA needed

    plt.xlabel("Number of IRS Elements", fontsize=12)
    plt.ylabel("Average Sum Rate (bps/Hz)", fontsize=12)
    plt.title(f"Avg Sum Rate vs IRS Elements: OMA (Analytical)", fontsize=14)
    plt.legend(fontsize=10, loc='upper left') # Adjusted legend
    plt.grid(True, linestyle='--')
    plt.xticks(elements)
    plt.yticks(fontsize=10)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.tight_layout()
    plt.show() # Show both plots (or the second one if first plt.show() was commented out)

else:
    print("Skipping Plot 3b (OMA): No valid OMA data generated.")


# --- End of Simulation 3 ---
end_time_3 = time.time()
print(f"\nSimulation 3 total time: {end_time_3 - start_time_3:.2f} seconds")