
# !!! Ensure IRSOMAEnv class is defined and compatible with DQNAgent !!!
# If IRSOMAEnv does not exist, this section will fail.

print("\nStarting Simulation 3: NOMA vs OMA Comparison (Averaged)...")
start_time_3 = time.time()

# Data structures for NOMA and OMA results
# Structure: {power_dbm: [ [run1_elem1, run1_elem2,...], [run2_elem1, run2_elem2,...], ... ]}
all_runs_noma_rates = {p_dbm: [] for p_dbm in powers}
all_runs_oma_rates = {p_dbm: [] for p_dbm in powers}

for run in range(num_runs):
    print(f"\n--- Run {run + 1} / {num_runs} --- (Plot 3)")
    # Results for this single run: {power_dbm: [elem1_rate, elem2_rate, ...]}
    run_noma_rates = {p_dbm: [] for p_dbm in powers}
    run_oma_rates = {p_dbm: [] for p_dbm in powers}

    # Iterate through power levels specified in dBm
    for p_dbm in powers:
        # <<<--- Convert power from dBm to linear scale ---<<<
        # Assuming dBm refers to milliwatts (mW). Adjust if it refers to Watts.
        p_linear = 10**(p_dbm / 10.0)
        # ---------------------------------------------------->>>
        print(f"  Power level: {p_dbm} dBm ({p_linear:.2f} linear)")

        element_rates_noma_this_power = []
        element_rates_oma_this_power = []

        # Iterate through number of IRS elements
        for n in elements:
            # print(f"    Number of IRS elements: {n}") # Verbose

            # --- NOMA Simulation ---
            # print(f"      Simulating NOMA...") # Verbose
            # Pass the LINEAR power value to the environment
            env_noma = IRSNOMAEnv(num_elements=n, signal_power=p_linear)
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
                if done: pass # Handle done state if needed
            avg_noma = total_reward_noma / steps_noma if steps_noma > 0 else 0
            element_rates_noma_this_power.append(avg_noma)

            # --- OMA Simulation ---
            try:
                # print(f"      Simulating OMA...") # Verbose
                # Pass the LINEAR power value to the environment
                env_oma = IRSOMAEnv(num_elements=n, signal_power=p_linear) # Assuming class exists and is corrected
                agent_oma = DQNAgent(env_oma.state_dim, env_oma.action_dim, **agent_params)

                state = env_oma.reset()
                total_reward_oma = 0
                steps_oma = 0
                for i in range(fixed_iterations):
                    action = agent_oma.select_action(state)
                    next_state, reward, done = env_oma.step(action)
                    agent_oma.store(state, action, reward, next_state, done)
                    agent_oma.train()
                    state = next_state
                    total_reward_oma += reward
                    steps_oma += 1
                    if done: pass # Handle done state if needed
                avg_oma = total_reward_oma / steps_oma if steps_oma > 0 else 0
                element_rates_oma_this_power.append(avg_oma)
            except NameError:
                 print("\n !!! WARNING: IRSOMAEnv class not defined or error during OMA simulation. Skipping OMA for this point. !!! \n")
                 # Append a placeholder if OMA fails
                 element_rates_oma_this_power.append(np.nan) # Use NaN if OMA part fails

        # Store results for this power level (using original dBm value as key)
        run_noma_rates[p_dbm] = element_rates_noma_this_power
        run_oma_rates[p_dbm] = element_rates_oma_this_power

    # Append results for the whole run (using original dBm value as key)
    for p_dbm in powers:
        all_runs_noma_rates[p_dbm].append(run_noma_rates[p_dbm])
        all_runs_oma_rates[p_dbm].append(run_oma_rates[p_dbm])


# --- Averaging Results Across Runs (Plot 3) ---
print("\n--- Averaging Results Across Runs (Plot 3) ---")
# Use original dBm power values as keys
final_avg_noma = {p_dbm: [] for p_dbm in powers}
final_std_noma = {p_dbm: [] for p_dbm in powers}
final_avg_oma = {p_dbm: [] for p_dbm in powers}
final_std_oma = {p_dbm: [] for p_dbm in powers}

for p_dbm in powers:
    # NOMA
    noma_rates_array = np.array(all_runs_noma_rates[p_dbm])
    final_avg_noma[p_dbm] = np.nanmean(noma_rates_array, axis=0).tolist() # Use nanmean
    final_std_noma[p_dbm] = np.nanstd(noma_rates_array, axis=0).tolist()  # Use nanstd
    # OMA
    oma_rates_array = np.array(all_runs_oma_rates[p_dbm])
    final_avg_oma[p_dbm] = np.nanmean(oma_rates_array, axis=0).tolist() # Use nanmean
    final_std_oma[p_dbm] = np.nanstd(oma_rates_array, axis=0).tolist()  # Use nanstd


# --- Plotting Averaged Results (Plot 3) ---
print("\nPlotting Averaged Results (Plot 3)...")
plt.figure(figsize=(10, 7))
markers = ['o', 's', '^', 'd', 'v', '*'] # Different markers for powers
linestyles = ['-', '--', ':', '-.'] # Different linestyles
colors = plt.cm.viridis(np.linspace(0, 1, len(powers))) # Color map

# Check if any valid (non-NaN) OMA data exists after averaging
has_valid_oma = not all(all(np.isnan(val) for val in final_avg_oma[p_dbm]) for p_dbm in powers)

for i, p_dbm in enumerate(powers):
    color = colors[i]
    marker = markers[i % len(markers)]

    # Plot NOMA
    plt.plot(elements, final_avg_noma[p_dbm],
             label=f"NOMA P={p_dbm}dBm", marker=marker,
             linestyle=linestyles[0], color=color, linewidth=2)
    # Optional: Add shaded region for NOMA std dev
    lower_noma = np.array(final_avg_noma[p_dbm]) - np.array(final_std_noma[p_dbm])
    upper_noma = np.array(final_avg_noma[p_dbm]) + np.array(final_std_noma[p_dbm])
    plt.fill_between(elements, lower_noma, upper_noma, alpha=0.1, color=color)

    # Plot OMA only if valid data exists
    if has_valid_oma and not all(np.isnan(final_avg_oma[p_dbm])):
         plt.plot(elements, final_avg_oma[p_dbm],
                 label=f"OMA P={p_dbm}dBm", marker=marker,
                 linestyle=linestyles[1], color=color, linewidth=2) # Dashed line for OMA
         # Optional: Add shaded region for OMA std dev
         lower_oma = np.array(final_avg_oma[p_dbm]) - np.array(final_std_oma[p_dbm])
         upper_oma = np.array(final_avg_oma[p_dbm]) + np.array(final_std_oma[p_dbm])
         # Ensure fill_between handles potential NaNs if only some points failed
         valid_oma_indices = ~np.isnan(final_avg_oma[p_dbm])
         if np.any(valid_oma_indices):
              plt.fill_between(np.array(elements)[valid_oma_indices],
                               lower_oma[valid_oma_indices],
                               upper_oma[valid_oma_indices],
                               alpha=0.1, color=color)


plt.xlabel("Number of IRS Elements", fontsize=12)
plt.ylabel("Average Sum Rate (bps/Hz)", fontsize=12)
plt.title(f"Avg Sum Rate vs IRS Elements: NOMA vs OMA (Averaged over {num_runs} runs)", fontsize=14)
# Adjust legend position if it overlaps data
plt.legend(fontsize=9, loc='best') # May need 'upper left' or manual adjustment
plt.grid(True, linestyle='--')
plt.xticks(elements)
plt.yticks(fontsize=10)
# Consider setting y-axis limits if needed, e.g., plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

end_time_3 = time.time()
print(f"\nSimulation 3 time: {end_time_3 - start_time_3:.2f} seconds")