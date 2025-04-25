from LSTM.lstm_predictor import *      
from CLUSTERING.kgmm_clustering import *
from DQN.agent import *
from DQN.noma_env import *

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


# ----------------- Plot Section ---------------------

import numpy as np
import matplotlib.pyplot as plt
import time # Optional: To time the simulation

# --- Assumed Imports ---
# Make sure the updated DQNAgent class and IRSNOMAEnv class are defined
# in this file or imported correctly.
# Example:
# from irs_noma_env import IRSNOMAEnv # If env is in irs_noma_env.py
# from dqn_optimizer import DQNAgent # If agent is in dqn_optimizer.py
# -----------------------

# --- Simulation Parameters ---
powers = [5, 10, 20, 30, 40, 50]
iterations_list = [100, 300, 500, 700, 1000]
num_runs = 10  # <<<--- Number of independent runs to average over

# --- Agent Hyperparameters (Set desired values here) ---
agent_params = {
    'lr': 1e-4,              # Example: Adjusted Learning rate
    'gamma': 0.99,
    'epsilon': 1.0,
    'decay': 0.995,
    'epsilon_min': 0.01,
    'target_update_freq': 100, # Example: Adjusted Target update frequency
    'buffer_size': 10000
}

# --- Data Structures to Store Results ---
# Stores results for all runs: {iteration_count: [ [run1_pow1, run1_pow2,...], [run2_pow1, run2_pow2,...], ... ]}
all_runs_avg_sum_rates = {iteration: [] for iteration in iterations_list}

print(f"Starting simulation: {num_runs} runs, Iterations: {iterations_list}, Powers: {powers}")
start_time = time.time()

# --- Main Simulation Loop (Over Multiple Runs) ---
for run in range(num_runs):
    print(f"\n--- Run {run + 1} / {num_runs} ---")
    # Stores results for this single run: {iteration_count: [pow1_rate, pow2_rate, ...]}
    run_avg_sum_rates = {iteration: [] for iteration in iterations_list}

    for iteration_count in iterations_list:
        print(f"  Iteration count: {iteration_count}")
        power_rates_this_iter = [] # Temporarily store rates for this iteration count before appending

        for p in powers:
            # print(f"    Simulating with transmit power = {p} dBm") # Verbose logging

            # Initialize environment and agent for EACH power level simulation
            env = IRSNOMAEnv(signal_power=p) # Assuming default env params are okay
            agent = DQNAgent(
                env.state_dim,
                env.action_dim,
                **agent_params # Pass hyperparameters defined above
            )

            state = env.reset()
            total_reward = 0
            steps_in_run = 0

            # Training loop for the specified number of iterations (steps)
            for i in range(iteration_count):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action) # env.step should return reward for the *current* step
                agent.store(state, action, reward, next_state, done)
                agent.train() # Perform training step
                state = next_state
                total_reward += reward
                steps_in_run += 1

                # Note: The original 'done' condition (time > 30) means episodes are very short.
                # The agent trains across these short episode boundaries.
                # If 'done' indicated a terminal state requiring reset, you'd handle it here.
                if done:
                    # print(f"    Episode ended early at step {i}") # Optional info
                    # state = env.reset() # Reset if needed by env logic, currently likely not needed
                    pass

            # Calculate average reward for this specific (power, iteration_count) run
            # Averaging reward achieved DURING training
            avg_reward_this_run = total_reward / steps_in_run if steps_in_run > 0 else 0
            # print(f"      Avg sum rate (Training): {avg_reward_this_run:.4f}") # Verbose logging
            power_rates_this_iter.append(avg_reward_this_run)


            # --- Alternative: Evaluate AFTER Training (Recommended for clearer performance view) ---
            # Comment out the calculation above and uncomment this section if you prefer evaluation
            # print(f"    Starting evaluation for power = {p} dBm...")
            # eval_steps = 100  # Number of steps to evaluate the learned policy
            # eval_total_reward = 0
            # eval_state = env.reset() # Use a fresh state for evaluation consistency
            # current_epsilon = agent.epsilon # Save agent's current epsilon state if needed
            # agent.epsilon = 0 # Use greedy policy (no exploration) for evaluation
            # eval_steps_done = 0
            # for _ in range(eval_steps):
            #     action = agent.select_action(eval_state)
            #     next_eval_state, reward, done = env.step(action)
            #     eval_total_reward += reward
            #     eval_state = next_eval_state
            #     eval_steps_done += 1
            #     if done:
            #         eval_state = env.reset() # Reset if eval episodes can end
            # agent.epsilon = current_epsilon # Restore agent's epsilon state if needed later
            # avg_reward_this_run = eval_total_reward / eval_steps_done if eval_steps_done > 0 else 0
            # print(f"      Avg EVALUATION sum rate: {avg_reward_this_run:.4f}")
            # power_rates_this_iter.append(avg_reward_this_run)
            # --- End Alternative ---


        # Store the list of rates for this iteration count for the current run
        run_avg_sum_rates[iteration_count] = power_rates_this_iter

    # Append the results of the entire run to the master list
    for iteration_count in iterations_list:
        all_runs_avg_sum_rates[iteration_count].append(run_avg_sum_rates[iteration_count])

# --- Averaging Results Across Runs ---
print("\n--- Averaging Results Across Runs ---")
final_avg_sum_rates = {iteration: [] for iteration in iterations_list}
final_std_sum_rates = {iteration: [] for iteration in iterations_list} # Optional: Standard deviation

for iteration_count in iterations_list:
    # Convert list of lists into a NumPy array for easy averaging
    # Shape will be (num_runs, num_powers)
    rates_array = np.array(all_runs_avg_sum_rates[iteration_count])

    # Calculate mean and std dev across the 'runs' axis (axis=0)
    final_avg_sum_rates[iteration_count] = np.mean(rates_array, axis=0).tolist()
    final_std_sum_rates[iteration_count] = np.std(rates_array, axis=0).tolist()
    # print(f"Iteration {iteration_count}: Avg Rates: {[f'{x:.4f}' for x in final_avg_sum_rates[iteration_count]]}")
    # print(f"Iteration {iteration_count}: Std Devs: {[f'{x:.4f}' for x in final_std_sum_rates[iteration_count]]}")


# --- Plotting Averaged Results ---
print("\nPlotting Averaged Results...")
plt.figure(figsize=(10, 7)) # Slightly larger figure
for iteration_count in iterations_list:
    # Plot the averaged line
    plt.plot(powers, final_avg_sum_rates[iteration_count],
             label=f"{iteration_count} Iterations", marker='o', linewidth=2)

    # Optional: Add shaded region for standard deviation
    lower_bound = np.array(final_avg_sum_rates[iteration_count]) - np.array(final_std_sum_rates[iteration_count])
    upper_bound = np.array(final_avg_sum_rates[iteration_count]) + np.array(final_std_sum_rates[iteration_count])
    plt.fill_between(powers, lower_bound, upper_bound, alpha=0.15) # Adjust alpha for transparency

    # Optional: Plot error bars instead of shaded region
    # plt.errorbar(powers, final_avg_sum_rates[iteration_count], yerr=final_std_sum_rates[iteration_count],
    #              label=f"{iteration_count} Iterations", marker='o', capsize=4, linestyle='-', linewidth=2)


plt.xlabel("Transmit Power (dBm)", fontsize=12)
plt.ylabel("Average Sum Rate (bps/Hz)", fontsize=12)
plt.title(f"Avg Sum Rate vs Transmit Power (Averaged over {num_runs} runs)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--')
plt.xticks(powers) # Ensure all power levels are marked
plt.yticks(fontsize=10)
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.show()

end_time = time.time()
print(f"\nTotal simulation time: {end_time - start_time:.2f} seconds")


import numpy as np
import matplotlib.pyplot as plt
import time

# --- Assumed Imports ---
# Make sure the updated DQNAgent class, IRSNOMAEnv class, and potentially
# an IRSOMAEnv class are defined/imported correctly.
# Example:
# from irs_noma_env import IRSNOMAEnv
# from irs_oma_env import IRSOMAEnv # Assuming OMA environment exists
# from dqn_optimizer import DQNAgent
# -----------------------

# --- Simulation Parameters ---
powers = [5, 10, 20, 30, 40, 50] # Use consistent power levels if needed across plots
elements = [5, 10, 15, 20, 25, 30]
num_runs = 10  # <<<--- Number of independent runs to average over
fixed_iterations = 300 # Fixed number of training steps for plots 2 and 3

# --- Agent Hyperparameters (Use the same consistent set) ---
agent_params = {
    'lr': 1e-4,
    'gamma': 0.99,
    'epsilon': 1.0,
    'decay': 0.995,
    'epsilon_min': 0.01,
    'target_update_freq': 100,
    'buffer_size': 10000
}

# ==============================================================================
# 2. Sum Rate vs IRS Elements with Different Powers (Averaged)
# ==============================================================================
print("\nStarting Simulation 2: Sum Rate vs IRS Elements (Averaged)...")
start_time_2 = time.time()

# Structure: {power: [ [run1_elem1, run1_elem2,...], [run2_elem1, run2_elem2,...], ... ]}
all_runs_avg_sum_rates_elements = {p: [] for p in powers}

for run in range(num_runs):
    print(f"\n--- Run {run + 1} / {num_runs} --- (Plot 2)")
    # Structure: {power: [elem1_rate, elem2_rate, ...]}
    run_avg_sum_rates_elements = {p: [] for p in powers}

    for p in powers:
        print(f"  Power level: {p} dBm")
        element_rates_this_power = [] # Temp list for rates at this power level

        for n in elements:
            # print(f"    Number of IRS elements: {n}") # Verbose
            env = IRSNOMAEnv(num_elements=n, signal_power=p)
            agent = DQNAgent(env.state_dim, env.action_dim, **agent_params)

            state = env.reset()
            total_reward = 0
            steps_in_run = 0

            for i in range(fixed_iterations): # Use fixed iterations
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.store(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                total_reward += reward
                steps_in_run += 1
                if done:
                    # print(f"    Episode ended early at step {i}") # Verbose
                    # state = env.reset() # Handle reset if needed
                    pass

            # Calculate average reward for this specific (element, power) run
            avg_reward_this_run = total_reward / steps_in_run if steps_in_run > 0 else 0
            element_rates_this_power.append(avg_reward_this_run)
            # print(f"      Elements={n}, Avg Rate={avg_reward_this_run:.4f}") # Verbose

        # Store the list of rates for this power level for the current run
        run_avg_sum_rates_elements[p] = element_rates_this_power

    # Append the results of the entire run to the master list
    for p in powers:
        all_runs_avg_sum_rates_elements[p].append(run_avg_sum_rates_elements[p])

# --- Averaging Results Across Runs (Plot 2) ---
print("\n--- Averaging Results Across Runs (Plot 2) ---")
final_avg_elements = {p: [] for p in powers}
final_std_elements = {p: [] for p in powers} # Optional: Standard deviation

for p in powers:
    rates_array = np.array(all_runs_avg_sum_rates_elements[p]) # Shape: (num_runs, num_elements)
    final_avg_elements[p] = np.mean(rates_array, axis=0).tolist()
    final_std_elements[p] = np.std(rates_array, axis=0).tolist()

# --- Plotting Averaged Results (Plot 2) ---
print("\nPlotting Averaged Results (Plot 2)...")
plt.figure(figsize=(10, 7))
for p in powers:
    plt.plot(elements, final_avg_elements[p],
             label=f"Power {p} dBm", marker='s', linewidth=2)
    # Optional: Add shaded region for standard deviation
    lower_bound = np.array(final_avg_elements[p]) - np.array(final_std_elements[p])
    upper_bound = np.array(final_avg_elements[p]) + np.array(final_std_elements[p])
    plt.fill_between(elements, lower_bound, upper_bound, alpha=0.15)

plt.xlabel("Number of IRS Elements", fontsize=12)
plt.ylabel("Average Sum Rate (bps/Hz)", fontsize=12)
plt.title(f"Avg Sum Rate vs IRS Elements (Averaged over {num_runs} runs)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--')
plt.xticks(elements)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

end_time_2 = time.time()
print(f"\nSimulation 2 time: {end_time_2 - start_time_2:.2f} seconds")


# ==============================================================================
# 3. Sum Rate vs IRS Elements (NOMA vs OMA) with Different Powers (Averaged)
# ==============================================================================

# ==============================================================================
# 3. Sum Rate vs IRS Elements: NOMA (RL) vs OMA (Analytical) (Averaged)
# ==============================================================================

print("\nStarting Simulation 3: NOMA (RL) vs OMA (Analytical) Comparison (Averaged)...")
start_time_3 = time.time()

# Data structures for NOMA (from RL) and OMA (analytical) results
all_runs_noma_rates = {p_dbm: [] for p_dbm in powers}
all_runs_oma_rates = {p_dbm: [] for p_dbm in powers} # Will store analytical results

for run in range(num_runs):
    print(f"\n--- Run {run + 1} / {num_runs} --- (Plot 3)")
    run_noma_rates = {p_dbm: [] for p_dbm in powers}
    run_oma_rates = {p_dbm: [] for p_dbm in powers} # Store analytical OMA per run

    for p_dbm in powers:
        p_linear = 10**(p_dbm / 10.0)
        print(f"  Power level: {p_dbm} dBm ({p_linear:.2f} linear)")

        element_rates_noma_this_power = []
        element_rates_oma_this_power = [] # Store analytical OMA results

        for n in elements:
            # --- NOMA Simulation (RL based - Keep as before) ---
            # print(f"    NOMA (RL): Elements={n}") # Verbose
            # Assuming IRSNOMAEnv uses num_users=num_users_for_comparison
            env_noma = IRSNOMAEnv(num_elements=n, signal_power=p_linear, num_users=10)
            agent_noma = DQNAgent(env_noma.state_dim, env_noma.action_dim, **agent_params)
            state = env_noma.reset()
            total_reward_noma = 0
            steps_noma = 0
            for i in range(fixed_iterations):
                action = agent_noma.select_action(state)
                next_state, reward, done = env_noma.step(action)
                agent_noma.store(state, action, reward, next_state, done)
                agent_noma.train()
                state = next_state
                total_reward_noma += reward
                steps_noma += 1
                if done: pass
            avg_noma = total_reward_noma / steps_noma if steps_noma > 0 else 0
            element_rates_noma_this_power.append(avg_noma)

            # --- OMA Analytical Calculation ---
            # print(f"    OMA (Analytical): Elements={n}") # Verbose

            # Assume ideal gain (phase alignment) for baseline OMA comparison
            gain_oma_analytical = 1.0
            scaled_gain_oma_analytical = gain_oma_analytical * (n ** 2)
            noise_oma = 1.0 # Consistent noise power
            interference_oma = 0.0 # Definition of OMA

            # Calculate the interference-free SNR for a single representative OMA user
            snr_oma_analytical = (scaled_gain_oma_analytical * p_linear) / (interference_oma + noise_oma + 1e-10)

            # Calculate Analytical OMA Sum Rate = log2(1 + SNR_single_user)
            avg_oma_analytical = np.log2(1 + snr_oma_analytical)
            element_rates_oma_this_power.append(avg_oma_analytical)
            # --- End OMA Analytical Calculation ---

        # Store results for this power level
        run_noma_rates[p_dbm] = element_rates_noma_this_power
        run_oma_rates[p_dbm] = element_rates_oma_this_power

    # Append results for the whole run
    for p_dbm in powers:
        all_runs_noma_rates[p_dbm].append(run_noma_rates[p_dbm])
        all_runs_oma_rates[p_dbm].append(run_oma_rates[p_dbm])

# --- Averaging Results Across Runs ---
# Averaging NOMA (RL results) smooths RL variance.
# Averaging OMA (analytical results) will just return the deterministic analytical value,
# but we keep the structure consistent.
print("\n--- Averaging Results Across Runs (Plot 3) ---")
final_avg_noma = {p_dbm: [] for p_dbm in powers}
final_std_noma = {p_dbm: [] for p_dbm in powers}
final_avg_oma = {p_dbm: [] for p_dbm in powers}
final_std_oma = {p_dbm: [] for p_dbm in powers} # Will be ~zero for analytical OMA

for p_dbm in powers:
    noma_rates_array = np.array(all_runs_noma_rates[p_dbm])
    final_avg_noma[p_dbm] = np.nanmean(noma_rates_array, axis=0).tolist()
    final_std_noma[p_dbm] = np.nanstd(noma_rates_array, axis=0).tolist()

    oma_rates_array = np.array(all_runs_oma_rates[p_dbm]) # Contains identical rows
    final_avg_oma[p_dbm] = np.nanmean(oma_rates_array, axis=0).tolist()
    final_std_oma[p_dbm] = np.nanstd(oma_rates_array, axis=0).tolist() # Should be near zero

# --- Plotting Averaged Results (Plot 3 - NOMA vs Analytical OMA) ---
print("\nPlotting Averaged Results (Plot 3: NOMA (RL) vs OMA (Analytical))...")
plt.figure(figsize=(10, 7))
markers = ['o', 's', '^', 'd', 'v', '*']
linestyles = ['-', '--', ':', '-.']
colors = plt.cm.viridis(np.linspace(0, 1, len(powers)))

# Check if any NOMA data is valid (should be unless sim failed)
has_valid_noma = not all(all(np.isnan(val) for val in final_avg_noma[p_dbm]) for p_dbm in powers)

if has_valid_noma: # Proceed only if NOMA results exist
    for i, p_dbm in enumerate(powers):
        color = colors[i]
        marker = markers[i % len(markers)]

        # Plot NOMA (RL Results)
        plt.plot(elements, final_avg_noma[p_dbm],
                 label=f"NOMA (RL) P={p_dbm}dBm", marker=marker,
                 linestyle=linestyles[1], color=color, linewidth=2)
        lower_noma = np.array(final_avg_noma[p_dbm]) - np.array(final_std_noma[p_dbm])
        upper_noma = np.array(final_avg_noma[p_dbm]) + np.array(final_std_noma[p_dbm])
        plt.fill_between(elements, lower_noma, upper_noma, alpha=0.15, color=color)

        # Plot OMA (Analytical Results)
        plt.plot(elements, final_avg_oma[p_dbm],
                 label=f"OMA (Analytical) P={p_dbm}dBm", marker=marker,
                 linestyle=linestyles[0], color=color, linewidth=2) # Dashed line for OMA
        # No shaded region for analytical OMA needed as std dev should be ~0


    plt.xlabel("Number of IRS Elements", fontsize=12)
    plt.ylabel("Average Sum Rate (bps/Hz)", fontsize=12)
    plt.title(f"Avg Sum Rate vs IRS Elements: NOMA (RL) vs OMA (Analytical) (NOMA Avg over {num_runs} runs)", fontsize=14)
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, linestyle='--')
    plt.xticks(elements)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping plot as no valid NOMA data was found.")


end_time_3_total = time.time()
# Note: Simulation time only reflects NOMA RL part now
print(f"\nSimulation 3 total time: {end_time_3_total - start_time_3:.2f} seconds")