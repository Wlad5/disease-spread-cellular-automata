"""
Infection Probability + Waning Probability Combined Sensitivity Analysis

Analyzes how disease dynamics (S, I, R) vary with both infection_prob and waning_prob.
Keeps mixing_rate = 0 and all other parameters constant.
Produces:
1. Time series comparison plots (CA vs ODE) for selected parameter combinations
2. Heatmaps of L2 norms for S, I, R showing how they vary across both parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import importlib.util
import csv
from scipy.interpolate import interp1d

# ==========================================================
# Dynamic imports of CA and ODE modules
# ==========================================================
ca_path     = os.path.join(os.path.dirname(__file__), "SIRS.py")
ode_path    = os.path.join(os.path.dirname(__file__), "SIRS_ODE_SOLVER.py")

# ==========================================================
# Output directories
# ==========================================================
def setup_output_directories(initial_infected_count):
    """Create CSV and images directories based on initial infected count."""
    csv_dir = f'infection_waning_combined_sensitivity_csv_infected_{initial_infected_count}'
    images_dir = f'infection_waning_combined_sensitivity_images_infected_{initial_infected_count}'
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    return csv_dir, images_dir

def import_module_from_path(module_name, file_path):
    spec    = importlib.util.spec_from_file_location(module_name, file_path)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

SIRS_CA     = import_module_from_path("SIRS_CA", ca_path)
SIRS_ODE    = import_module_from_path("SIRS_ODE", ode_path)

# ==========================================================
# Default parameters (fixed)
# ==========================================================
params = {
    'recovery_prob'         : 0.1,
    'k'                     : 8,
    'delta_t'               : 1.0,
    't_max'                 : 500,
    'dt'                    : 1.0,
    'ode_dt'                : 0.1,
    'cell_size'             : 4,
    'initial_infected_count': 100,
    'width'                 : 10,
    'height'                : 10,
    'mixing_rate'           : 0.0,  # No mixing
    'num_simulations'       : 1
}

# Parameter ranges for combined analysis
infection_prob_values = np.linspace(0.1, 0.5, 20)
waning_prob_values = np.linspace(0.1, 0.5, 20)

# ==========================================================
# Helper: compute initial infected fraction dynamically
# ==========================================================
def compute_initial_infected_fraction(width, height, initial_infected_count):
    """Compute the true fraction of infected population for ODE."""
    total_cells = width * height
    return initial_infected_count / total_cells

# ==========================================================
# Simulation runners
# ==========================================================
def run_ca_simulation(infection_prob, recovery_prob, waning_prob, t_max,
                      width, height, cell_size, initial_infected_count, mixing_rate=0.0):
    """Run CA simulation and return history + initial infected fraction."""
    sim = SIRS_CA.Simulation(
        width           =width,
        height          =height,
        cell_size       =cell_size,
        infection_prob  =infection_prob,
        recovery_prob   =recovery_prob,
        waning_prob     =waning_prob,
        delta_t         =1,
        mixing_rate     =mixing_rate
    )

    # Infect a given number of random cells
    sim.grid.infect_random(n=initial_infected_count)

    # Compute actual initial infected fraction (used by ODE)
    initial_infected_frac = compute_initial_infected_fraction(width, height, initial_infected_count)

    # Run CA simulation steps (without GUI)
    sim.running = False
    for _ in range(t_max):
        sim.grid.update()

    history = sim.get_history()
    return history, initial_infected_frac

def run_ode_simulation(infection_prob, recovery_prob, waning_prob, k, delta_t,
                       initial_infected_frac, t_max, ode_dt):
    """Run ODE simulation using parameters mapped from CA."""
    time_points, states, _ = SIRS_ODE.solve_sirs_from_ca_params(
        infection_prob      =infection_prob,
        recovery_prob       =recovery_prob,
        waning_prob         =waning_prob,
        k                   =k,
        delta_t             =delta_t,
        initial_infected    =initial_infected_frac,
        t_max               =t_max,
        dt                  =ode_dt
    )
    return time_points, states

# ==========================================================
# Calculate norms
# ==========================================================
def calculate_norm(ca_history, ode_time, ode_states):
    """
    Calculate L2 norms between CA and ODE for S, I, R.
    
    Returns:
        dict with 'S', 'I', 'R' keys, each containing the L2 norm value
    """
    ca_timesteps = np.array(ca_history['timestep'])
    
    # Interpolate ODE results to match CA time steps
    ode_time_interp = np.linspace(0, len(ode_states) * params['ode_dt'], len(ode_states))
    interp_ode_S = interp1d(ode_time_interp, ode_states[:, 0], kind='linear', fill_value="extrapolate")
    interp_ode_I = interp1d(ode_time_interp, ode_states[:, 1], kind='linear', fill_value="extrapolate")
    interp_ode_R = interp1d(ode_time_interp, ode_states[:, 2], kind='linear', fill_value="extrapolate")
    
    # Calculate L2 norms
    norm_S = np.sqrt(np.sum((np.array(ca_history['S_frac']) - interp_ode_S(ca_timesteps)) ** 2))
    norm_I = np.sqrt(np.sum((np.array(ca_history['I_frac']) - interp_ode_I(ca_timesteps)) ** 2))
    norm_R = np.sqrt(np.sum((np.array(ca_history['R_frac']) - interp_ode_R(ca_timesteps)) ** 2))
    
    return {'S': norm_S, 'I': norm_I, 'R': norm_R}

# ==========================================================
# Heatmap plots
# ==========================================================
def plot_norm_heatmaps(norms_S, norms_I, norms_R, infection_labels, waning_labels, images_dir):
    """
    Plot heatmaps for L2 norms of S, I, R across infection_prob and waning_prob.
    Creates 3 separate figures, one for each state (S, I, R).
    Parameters start from smallest values in bottom left corner.
    """
    # Prepare data for heatmaps - flip vertically so smallest infection prob is at bottom
    data_S = np.array(norms_S)[::-1, :]
    data_I = np.array(norms_I)[::-1, :]
    data_R = np.array(norms_R)[::-1, :]
    
    # Flip labels to match the flipped data
    infection_labels_flipped = infection_labels[::-1]
    
    # Plot heatmap for S
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data_S, xticklabels=waning_labels, yticklabels=infection_labels_flipped,
                cmap='YlOrRd', ax=ax, cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.2f', annot_kws={'size': 6})
    ax.set_title('L2 Norm: Susceptible (S)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Waning Probability', fontsize=12)
    ax.set_ylabel('Infection Probability', fontsize=12)
    plt.tight_layout()
    heatmap_path = os.path.join(images_dir, 'heatmap_S_infection_waning_combined.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap for S saved to '{heatmap_path}'")
    plt.close()
    
    # Plot heatmap for I
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data_I, xticklabels=waning_labels, yticklabels=infection_labels_flipped,
                cmap='YlOrRd', ax=ax, cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.2f', annot_kws={'size': 6})
    ax.set_title('L2 Norm: Infected (I)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Waning Probability', fontsize=12)
    ax.set_ylabel('Infection Probability', fontsize=12)
    plt.tight_layout()
    heatmap_path = os.path.join(images_dir, 'heatmap_I_infection_waning_combined.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap for I saved to '{heatmap_path}'")
    plt.close()
    
    # Plot heatmap for R
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data_R, xticklabels=waning_labels, yticklabels=infection_labels_flipped,
                cmap='YlOrRd', ax=ax, cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.2f', annot_kws={'size': 6})
    ax.set_title('L2 Norm: Recovered (R)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Waning Probability', fontsize=12)
    ax.set_ylabel('Infection Probability', fontsize=12)
    plt.tight_layout()
    heatmap_path = os.path.join(images_dir, 'heatmap_R_infection_waning_combined.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap for R saved to '{heatmap_path}'")
    plt.close()

# ==========================================================
# Helper plotting functions for intermediate results
# ==========================================================
def _save_timeseries_to_csv(waning_results, infection_prob, waning_probs, experiment_name, csv_dir):
    """Save time series data with mean and std deviation to CSV files."""
    for waning_idx, waning_prob in enumerate(waning_probs):
        ca_histories, ode_time, ode_states_mean = waning_results[waning_idx]
        
        filename = f'{experiment_name}_infection_{infection_prob:.4f}_waning_{waning_prob:.6f}.csv'
        filepath = os.path.join(csv_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'CA_S_mean', 'CA_S_std', 'CA_I_mean', 'CA_I_std', 'CA_R_mean', 'CA_R_std', 
                           'ODE_S', 'ODE_I', 'ODE_R'])
            
            ca_timesteps = np.array(ca_histories[0]['timestep'])
            
            # Calculate mean and std for all CA simulations
            ca_S_all = np.array([h['S_frac'] for h in ca_histories])
            ca_I_all = np.array([h['I_frac'] for h in ca_histories])
            ca_R_all = np.array([h['R_frac'] for h in ca_histories])
            
            ca_S_mean = np.mean(ca_S_all, axis=0)
            ca_S_std = np.std(ca_S_all, axis=0)
            ca_I_mean = np.mean(ca_I_all, axis=0)
            ca_I_std = np.std(ca_I_all, axis=0)
            ca_R_mean = np.mean(ca_R_all, axis=0)
            ca_R_std = np.std(ca_R_all, axis=0)
            
            # Interpolate ODE to match CA timesteps
            ode_time_interp = np.linspace(0, len(ode_states_mean) * params['ode_dt'], len(ode_states_mean))
            interp_ode_S = interp1d(ode_time_interp, ode_states_mean[:, 0], kind='linear', fill_value="extrapolate")
            interp_ode_I = interp1d(ode_time_interp, ode_states_mean[:, 1], kind='linear', fill_value="extrapolate")
            interp_ode_R = interp1d(ode_time_interp, ode_states_mean[:, 2], kind='linear', fill_value="extrapolate")
            
            for i, t in enumerate(ca_timesteps):
                s_ode = interp_ode_S(t)
                i_ode = interp_ode_I(t)
                r_ode = interp_ode_R(t)
                writer.writerow([t, ca_S_mean[i], ca_S_std[i], ca_I_mean[i], ca_I_std[i], 
                               ca_R_mean[i], ca_R_std[i], s_ode, i_ode, r_ode])

# ==========================================================
# Helper plotting functions for intermediate results
# ==========================================================
def _plot_partial_heatmaps(norms_S, norms_I, norms_R, infection_labels_partial, waning_labels, param_idx, total_params, images_dir):
    """Plot partial heatmaps with data collected so far."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Prepare data for heatmaps
    data_S = np.array(norms_S)
    data_I = np.array(norms_I)
    data_R = np.array(norms_R)
    
    # Plot heatmap for S
    sns.heatmap(data_S, xticklabels=waning_labels, yticklabels=infection_labels_partial,
                cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f', vmin=0, vmax=0.5)
    axes[0].set_title('L2 Norm: Susceptible (S)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Waning Probability', fontsize=11)
    axes[0].set_ylabel('Infection Probability', fontsize=11)
    
    # Plot heatmap for I
    sns.heatmap(data_I, xticklabels=waning_labels, yticklabels=infection_labels_partial,
                cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f', vmin=0, vmax=0.5)
    axes[1].set_title('L2 Norm: Infected (I)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Waning Probability', fontsize=11)
    axes[1].set_ylabel('Infection Probability', fontsize=11)
    
    # Plot heatmap for R
    sns.heatmap(data_R, xticklabels=waning_labels, yticklabels=infection_labels_partial,
                cmap='YlOrRd', ax=axes[2], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f', vmin=0, vmax=0.5)
    axes[2].set_title('L2 Norm: Recovered (R)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Waning Probability', fontsize=11)
    axes[2].set_ylabel('Infection Probability', fontsize=11)
    
    plt.tight_layout()
    heatmap_path = os.path.join(images_dir, f'heatmaps_partial_infection{param_idx+1}of{total_params}.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"  → Partial heatmap saved")
    plt.close()

def _plot_timeseries_for_infection(waning_results, infection_prob, waning_probs, experiment_name, images_dir, csv_dir):
    """
    Plot time series for all waning_prob values of a given infection_prob.
    Organization: 3 rows (S, I, R states) × 5 columns (waning_prob values per batch)
    Each cell shows CA vs ODE curves with uncertainty bands.
    Plots are generated in batches of 5 waning probabilities per figure.
    """
    num_waning = len(waning_probs)
    plots_per_row = 5
    
    # Row 0: S (Susceptible), Row 1: I (Infected), Row 2: R (Recovered)
    states = [
        (0, 'S', 'blue'),
        (1, 'I', 'red'),
        (2, 'R', 'green')
    ]
    
    # Calculate mean and std for all CA simulations once
    ca_S_all = np.array([h['S_frac'] for h in waning_results[0][0]])
    ca_I_all = np.array([h['I_frac'] for h in waning_results[0][0]])
    ca_R_all = np.array([h['R_frac'] for h in waning_results[0][0]])
    
    ca_means_template = [np.mean(ca_S_all, axis=0), np.mean(ca_I_all, axis=0), np.mean(ca_R_all, axis=0)]
    ca_stds_template = [np.std(ca_S_all, axis=0), np.std(ca_I_all, axis=0), np.std(ca_R_all, axis=0)]
    
    # Create batches
    num_batches = (num_waning + plots_per_row - 1) // plots_per_row
    
    for batch_num in range(num_batches):
        start_idx = batch_num * plots_per_row
        end_idx = min(start_idx + plots_per_row, num_waning)
        batch_size = end_idx - start_idx
        
        # Create figure with 3 rows and batch_size columns
        fig, axes = plt.subplots(3, batch_size, figsize=(5 * batch_size, 12))
        
        # Handle case with single column
        if batch_size == 1:
            axes = axes.reshape(3, 1)
        
        for row, (state_idx, state_name, color) in enumerate(states):
            for batch_col, waning_idx in enumerate(range(start_idx, end_idx)):
                ax = axes[row, batch_col]
                waning_prob = waning_probs[waning_idx]
                
                ca_histories, ode_time, ode_states = waning_results[waning_idx]
                ca_timesteps = np.array(ca_histories[0]['timestep'])
                
                # Calculate mean and std for CA
                ca_S_all = np.array([h['S_frac'] for h in ca_histories])
                ca_I_all = np.array([h['I_frac'] for h in ca_histories])
                ca_R_all = np.array([h['R_frac'] for h in ca_histories])
                
                ca_means = [np.mean(ca_S_all, axis=0), np.mean(ca_I_all, axis=0), np.mean(ca_R_all, axis=0)]
                ca_stds = [np.std(ca_S_all, axis=0), np.std(ca_I_all, axis=0), np.std(ca_R_all, axis=0)]
                
                # CA plot with confidence band
                ax.plot(ca_timesteps, ca_means[state_idx],
                       linestyle='--', color=color, alpha=0.8, linewidth=2.5, label=f'CA {state_name} (mean)')
                ax.fill_between(ca_timesteps, 
                                ca_means[state_idx] - ca_stds[state_idx],
                                ca_means[state_idx] + ca_stds[state_idx],
                                color=color, alpha=0.2, label=f'CA {state_name} (±1 std)')
                
                # ODE plot
                ax.plot(ode_time, ode_states[:, state_idx], linestyle='-', color=color, linewidth=2.5, label=f'ODE {state_name}')
                
                ax.set_xlabel('Time', fontsize=10)
                ax.set_ylabel('Fraction', fontsize=10)
                ax.set_title(f'{state_name} - Waning: {waning_prob:.4f}', fontsize=11, fontweight='bold')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
        
        # Add a main title for the batch
        if num_batches > 1:
            batch_label = f"Batch {batch_num + 1}/{num_batches}"
        else:
            batch_label = ""
        
        fig.suptitle(f'Time Series for Infection Prob: {infection_prob:.3f} (varying waning prob) - {batch_label}', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        # Save plot
        if num_batches > 1:
            plot_filename = f'{experiment_name}_infection_{infection_prob:.4f}_batch{batch_num + 1}.png'
        else:
            plot_filename = f'{experiment_name}_infection_{infection_prob:.4f}.png'
        
        plot_path = os.path.join(images_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  → Time series plot saved: {plot_path}")
        plt.close()
    
    # Save CSV data
    _save_timeseries_to_csv(waning_results, infection_prob, waning_probs, experiment_name, csv_dir)

# ==========================================================
# Main experiment
# ==========================================================
def infection_waning_combined_sensitivity_experiment():
    # Setup output directories
    csv_dir, images_dir = setup_output_directories(params['initial_infected_count'])
    
    print("="*70)
    print("Infection Probability + Waning Probability Combined Sensitivity Analysis")
    print("="*70)
    print(f"\nInfection Probability values: {infection_prob_values}")
    print(f"Waning Probability values: {waning_prob_values}")
    print(f"Simulations per combination: {params['num_simulations']}")
    print(f"Mixing Rate: {params['mixing_rate']} (fixed)")
    print(f"Recovery Probability: {params['recovery_prob']} (fixed)")
    print(f"Initial Infected Count: {params['initial_infected_count']}")
    print(f"Output directories:")
    print(f"  CSV: {csv_dir}")
    print(f"  Images: {images_dir}")
    print()
    
    # Store results: results[infection_idx][waning_idx] = (ca_history, ode_time, ode_states)
    all_results = []
    
    # Store norms for heatmap
    norms_S = []
    norms_I = []
    norms_R = []
    
    initial_infected_frac = compute_initial_infected_fraction(
        params['width'], params['height'], params['initial_infected_count']
    )
    
    # Iterate over infection probabilities
    for infection_idx, infection_prob in enumerate(infection_prob_values):
        total_cells = params['width'] * params['height']
        print(f"\n{'='*70}")
        print(f"Infection Probability: {infection_prob:.4f}")
        print(f"{'='*70}")
        
        norms_S_row = []
        norms_I_row = []
        norms_R_row = []
        
        waning_results = []
        
        # Iterate over waning probabilities
        for waning_idx, waning_prob in enumerate(waning_prob_values):
            print(f"  [{waning_idx+1}/{len(waning_prob_values)}] Waning prob: {waning_prob:.6f}...", end=' ', flush=True)
            
            ca_histories = []
            ode_states_list = []
            ode_time = None
            
            # Run multiple simulations
            for sim_num in range(params['num_simulations']):
                ca_history, initial_infected = run_ca_simulation(
                    infection_prob=infection_prob,
                    recovery_prob=params['recovery_prob'],
                    waning_prob=waning_prob,
                    t_max=params['t_max'],
                    width=params['width'],
                    height=params['height'],
                    cell_size=params['cell_size'],
                    initial_infected_count=params['initial_infected_count'],
                    mixing_rate=params['mixing_rate']
                )
                ca_histories.append(ca_history)
                
                ode_time, ode_states = run_ode_simulation(
                    infection_prob=infection_prob,
                    recovery_prob=params['recovery_prob'],
                    waning_prob=waning_prob,
                    k=params['k'],
                    delta_t=params['delta_t'],
                    initial_infected_frac=initial_infected_frac,
                    t_max=params['t_max'],
                    ode_dt=params['ode_dt']
                )
                ode_states_list.append(ode_states)
            
            # Calculate mean CA history
            ca_timesteps = np.array(ca_histories[0]['timestep'])
            ca_S_mean = np.mean([h['S_frac'] for h in ca_histories], axis=0)
            ca_I_mean = np.mean([h['I_frac'] for h in ca_histories], axis=0)
            ca_R_mean = np.mean([h['R_frac'] for h in ca_histories], axis=0)
            
            ca_history_mean = {
                'timestep': ca_timesteps,
                'S_frac': ca_S_mean,
                'I_frac': ca_I_mean,
                'R_frac': ca_R_mean
            }
            
            # Calculate mean ODE states
            ode_states_mean = np.mean(ode_states_list, axis=0)
            
            # Calculate norms
            norms = calculate_norm(ca_history_mean, ode_time, ode_states_mean)
            norms_S_row.append(norms['S'])
            norms_I_row.append(norms['I'])
            norms_R_row.append(norms['R'])
            
            waning_results.append((ca_histories, ode_time, ode_states_mean))
            print("✓")
        
        all_results.append(waning_results)
        norms_S.append(norms_S_row)
        norms_I.append(norms_I_row)
        norms_R.append(norms_R_row)
        
        # PLOT AFTER EACH INFECTION PROBABILITY COMPLETES
        print(f"\nGenerating plots for infection prob {infection_prob:.4f}...")
        
        # Plot time series for this infection probability and save CSV data
        _plot_timeseries_for_infection(waning_results, infection_prob, waning_prob_values,
                                       'INFECTION_WANING_COMBINED_SENSITIVITY', images_dir, csv_dir)
        
        print(f"Plots and data saved for infection prob {infection_prob:.4f}\n")
    
    # Create labels for final heatmaps
    infection_labels = [f"{p:.3f}" for p in infection_prob_values]
    waning_labels = [f"{p:.4f}" for p in waning_prob_values]
    
    # Plot final complete heatmaps
    print("\n" + "="*70)
    print("Generating final complete heatmaps...")
    plot_norm_heatmaps(norms_S, norms_I, norms_R, infection_labels, waning_labels, images_dir)
    
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70)

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    infection_waning_combined_sensitivity_experiment()
