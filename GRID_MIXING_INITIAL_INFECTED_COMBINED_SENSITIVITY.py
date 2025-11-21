"""
Grid Size, Mixing Rate, and Initial Infected Count Combined Sensitivity Analysis

Analyzes how disease dynamics (S, I, R) vary with grid size, while mixing rate and 
initial infected count are varied together.

Produces:
1. Time series comparison plots (CA vs ODE) for each grid size with varying mixing/initial infected
2. Heatmaps of L2 norms for S, I, R showing how they vary across both parameters
3. CSV files with results for each parameter combination
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

def import_module_from_path(module_name, file_path):
    spec    = importlib.util.spec_from_file_location(module_name, file_path)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

SIRS_CA     = import_module_from_path("SIRS_CA", ca_path)
SIRS_ODE    = import_module_from_path("SIRS_ODE", ode_path)

# ==========================================================
# Directory setup
# ==========================================================
def setup_directories(base_name):
    """Create output directories for CSV and images"""
    csv_dir = os.path.join(os.path.dirname(__file__), f"{base_name}_csv")
    img_dir = os.path.join(os.path.dirname(__file__), f"{base_name}_images")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    return csv_dir, img_dir

csv_dir, img_dir = setup_directories("grid_mixing_initial_infected")

# ==========================================================
# Default parameters
# ==========================================================
params = {
    'infection_prob'        : 0.08,
    'recovery_prob'         : 0.1,
    'waning_prob'           : 0.002,
    'k'                     : 8,
    'delta_t'               : 1.0,
    't_max'                 : 500,
    'dt'                    : 1.0,
    'ode_dt'                : 0.1,
    'cell_size'             : 4,
    'num_simulations'       : 2
}

# Parameter ranges for combined analysis
grid_sizes = [(50, 50), (100, 100), (150, 150), (200, 200), (300, 300)]
# grid_sizes = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]

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
                      width, height, cell_size, initial_infected_count, mixing_rate=0.05):
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
    Calculate L2 norms between CA mean and ODE mean for S, I, R.
    
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
def plot_norm_heatmaps(norms_S, norms_I, norms_R, grid_labels, mixing_initial_labels):
    """
    Plot heatmaps for L2 norms of S, I, R across grid sizes and mixing/initial infected pairs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Prepare data for heatmaps (reverse rows so smallest grid size is at bottom)
    data_S = np.array(norms_S)[::-1]
    data_I = np.array(norms_I)[::-1]
    data_R = np.array(norms_R)[::-1]
    
    # Reverse grid labels to match reversed data
    grid_labels_reversed = list(reversed(grid_labels))
    
    # Plot heatmap for S
    sns.heatmap(data_S, xticklabels=mixing_initial_labels, yticklabels=grid_labels_reversed,
                cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f')
    axes[0].set_title('L2 Norm: Susceptible (S)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Mixing Rate / Initial Infected', fontsize=11)
    axes[0].set_ylabel('Grid Size', fontsize=11)
    
    # Plot heatmap for I
    sns.heatmap(data_I, xticklabels=mixing_initial_labels, yticklabels=grid_labels_reversed,
                cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f')
    axes[1].set_title('L2 Norm: Infected (I)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Mixing Rate / Initial Infected', fontsize=11)
    axes[1].set_ylabel('Grid Size', fontsize=11)
    
    # Plot heatmap for R
    sns.heatmap(data_R, xticklabels=mixing_initial_labels, yticklabels=grid_labels_reversed,
                cmap='YlOrRd', ax=axes[2], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f')
    axes[2].set_title('L2 Norm: Recovered (R)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Mixing Rate / Initial Infected', fontsize=11)
    axes[2].set_ylabel('Grid Size', fontsize=11)
    
    plt.tight_layout()
    filename = os.path.join(img_dir, 'heatmaps_norms_grid_mixing_initial_infected.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Heatmap plot saved to '{filename}'")
    plt.close()

# ==========================================================
# Helper plotting functions for intermediate results
# ==========================================================
def _plot_partial_heatmaps(norms_S, norms_I, norms_R, grid_labels_partial, mixing_initial_labels, grid_idx, total_grids):
    """Plot partial heatmaps with data collected so far."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Prepare data for heatmaps (reverse rows so smallest grid size is at bottom)
    data_S = np.array(norms_S)[::-1]
    data_I = np.array(norms_I)[::-1]
    data_R = np.array(norms_R)[::-1]
    
    # Reverse grid labels to match reversed data
    grid_labels_reversed = list(reversed(grid_labels_partial))
    
    # Plot heatmap for S
    sns.heatmap(data_S, xticklabels=mixing_initial_labels, yticklabels=grid_labels_reversed,
                cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f', vmin=0, vmax=0.5)
    axes[0].set_title('L2 Norm: Susceptible (S)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Mixing Rate / Initial Infected', fontsize=11)
    axes[0].set_ylabel('Grid Size', fontsize=11)
    
    # Plot heatmap for I
    sns.heatmap(data_I, xticklabels=mixing_initial_labels, yticklabels=grid_labels_reversed,
                cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f', vmin=0, vmax=0.5)
    axes[1].set_title('L2 Norm: Infected (I)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Mixing Rate / Initial Infected', fontsize=11)
    axes[1].set_ylabel('Grid Size', fontsize=11)
    
    # Plot heatmap for R
    sns.heatmap(data_R, xticklabels=mixing_initial_labels, yticklabels=grid_labels_reversed,
                cmap='YlOrRd', ax=axes[2], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f', vmin=0, vmax=0.5)
    axes[2].set_title('L2 Norm: Recovered (R)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Mixing Rate / Initial Infected', fontsize=11)
    axes[2].set_ylabel('Grid Size', fontsize=11)
    
    plt.tight_layout()
    filename = os.path.join(img_dir, f'heatmaps_partial_grid{grid_idx+1}of{total_grids}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  → Partial heatmap saved")
    plt.close()

def _plot_timeseries_for_grid(mixing_results, width, height, mixing_initial_pairs):
    """
    Plot time series for all mixing/initial infected pairs of a given grid size.
    Organization: 3 rows (S, I, R states) × N columns (mixing/initial infected pairs)
    Each cell shows CA vs ODE curves for that state and pair with std deviation shading.
    """
    num_pairs = len(mixing_initial_pairs)
    fig, axes = plt.subplots(3, num_pairs, figsize=(5 * num_pairs, 12))
    
    # Handle case with single pair
    if num_pairs == 1:
        axes = axes.reshape(3, 1)
    
    # Row 0: S (Susceptible), Row 1: I (Infected), Row 2: R (Recovered)
    states = [
        (0, 'S', 'blue'),
        (1, 'I', 'red'),
        (2, 'R', 'green')
    ]
    
    for row, (state_idx, state_name, color) in enumerate(states):
        for col, (pair_idx, (mixing_rate, initial_infected)) in enumerate(zip(range(num_pairs), mixing_initial_pairs)):
            ax = axes[row, col]
            
            ca_history, ode_time, ode_states, ode_states_std = mixing_results[pair_idx]
            
            # CA plot with shaded std deviation
            timesteps = ca_history['timestep']
            ca_mean = [ca_history['S_frac'], ca_history['I_frac'], ca_history['R_frac']][state_idx]
            ca_std = [ca_history['S_std'], ca_history['I_std'], ca_history['R_std']][state_idx]
            
            ax.plot(timesteps, ca_mean,
                   linestyle='--', color=color, alpha=0.8, linewidth=2, label=f'CA {state_name}')
            ax.fill_between(timesteps, ca_mean - ca_std, ca_mean + ca_std,
                           color=color, alpha=0.15, label=f'CA ±1 std')
            
            # ODE plot with shaded std deviation
            ax.plot(ode_time, ode_states[:, state_idx], linestyle='-', color=color, linewidth=2, label=f'ODE {state_name}')
            ax.fill_between(ode_time, ode_states[:, state_idx] - ode_states_std[:, state_idx],
                           ode_states[:, state_idx] + ode_states_std[:, state_idx],
                           color=color, alpha=0.1, label=f'ODE ±1 std')
            
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Fraction', fontsize=10)
            ax.set_title(f'{state_name} - Mix: {mixing_rate:.2f}, Init: {int(initial_infected)}', 
                        fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
    
    # Add a main title for the entire figure
    fig.suptitle(f'Time Series for Grid: {width}x{height} (varying mixing rate & initial infected)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    filename = os.path.join(img_dir, f'timeseries_grid_{width}x{height}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  → Time series saved: {filename}")
    plt.close()

# ==========================================================
# Save results to CSV
# ==========================================================
def save_results_to_csv(grid_idx, width, height, pair_idx, mixing_rate, initial_infected_count, 
                        ca_results, ode_results):
    """Save CA and ODE results to CSV files"""
    
    mean_ca, std_ca = ca_results
    ode_time, mean_ode_states, std_ode_states = ode_results
    
    # CA results
    ca_filename = os.path.join(csv_dir, f"ca_grid_{width}x{height}_pair_{pair_idx}.csv")
    with open(ca_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestep', 'S_frac', 'S_std', 'I_frac', 'I_std', 'R_frac', 'R_std'])
        for i in range(len(mean_ca['timestep'])):
            writer.writerow([
                mean_ca['timestep'][i],
                mean_ca['S_frac'][i], std_ca['S_frac'][i],
                mean_ca['I_frac'][i], std_ca['I_frac'][i],
                mean_ca['R_frac'][i], std_ca['R_frac'][i]
            ])
    
    # ODE results
    ode_filename = os.path.join(csv_dir, f"ode_grid_{width}x{height}_pair_{pair_idx}.csv")
    with open(ode_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'S', 'S_std', 'I', 'I_std', 'R', 'R_std'])
        for i in range(len(ode_time)):
            writer.writerow([
                ode_time[i],
                mean_ode_states[i, 0], std_ode_states[i, 0],
                mean_ode_states[i, 1], std_ode_states[i, 1],
                mean_ode_states[i, 2], std_ode_states[i, 2]
            ])

# ==========================================================
# Main experiment
# ==========================================================
def grid_mixing_initial_infected_combined_sensitivity_experiment():
    print("="*70)
    print("Grid Size, Mixing Rate, and Initial Infected Count Combined Analysis")
    print("="*70)
    print(f"\nGrid sizes: {grid_sizes}")
    print(f"Simulations per combination: {params['num_simulations']}")
    print()
    
    # Store results: grid_mixing_results[grid_idx][pair_idx] = (ca_history, ode_time, ode_states)
    grid_mixing_results = []
    
    # Store norms for heatmap
    norms_S = []
    norms_I = []
    norms_R = []
    
    # Iterate over grid sizes
    for grid_idx, (width, height) in enumerate(grid_sizes):
        total_cells = width * height
        print(f"\n{'='*70}")
        print(f"Grid: {width}x{height} ({total_cells} cells)")
        print(f"{'='*70}")
        
        # Create dynamic linspace based on total_cells for this grid size
        mixing_rate_range = np.linspace(0.0, 1.0, 6)
        initial_infected_count_range = np.linspace(5, total_cells, 6)
        mixing_initial_infected_pairs = list(zip(mixing_rate_range, initial_infected_count_range))
        print(f"Mixing/Initial Infected pairs: {mixing_initial_infected_pairs}")
        
        norms_S_row = []
        norms_I_row = []
        norms_R_row = []
        
        mixing_results = []
        
        # Iterate over mixing/initial infected pairs
        for pair_idx, (mixing_rate, initial_infected_count) in enumerate(mixing_initial_infected_pairs):
            initial_infected_count = int(initial_infected_count)
            print(f"  [{pair_idx+1}/{len(mixing_initial_infected_pairs)}] Mixing: {mixing_rate:.2f}, Init: {initial_infected_count}...", 
                  end=' ', flush=True)
            
            ca_histories = []
            ode_states_list = []
            ode_time = None
            
            # Run multiple simulations
            for sim_num in range(params['num_simulations']):
                print(f"s{sim_num+1}", end=' ', flush=True)
                ca_hist, ca_initial_infected_frac = run_ca_simulation(
                    infection_prob          =params['infection_prob'],
                    recovery_prob           =params['recovery_prob'],
                    waning_prob             =params['waning_prob'],
                    t_max                   =params['t_max'],
                    width                   =width,
                    height                  =height,
                    cell_size               =params['cell_size'],
                    initial_infected_count  =initial_infected_count,
                    mixing_rate             =mixing_rate
                )
                
                ode_time, ode_states = run_ode_simulation(
                    infection_prob          =params['infection_prob'],
                    recovery_prob           =params['recovery_prob'],
                    waning_prob             =params['waning_prob'],
                    k                       =params['k'],
                    delta_t                 =params['delta_t'],
                    initial_infected_frac   =ca_initial_infected_frac,
                    t_max                   =params['t_max'],
                    ode_dt                  =params['ode_dt']
                )
                
                ca_histories.append(ca_hist)
                ode_states_list.append(ode_states)
            
            # Calculate mean and std for CA results
            ca_S_values = np.array([np.array(hist['S_frac']) for hist in ca_histories])
            ca_I_values = np.array([np.array(hist['I_frac']) for hist in ca_histories])
            ca_R_values = np.array([np.array(hist['R_frac']) for hist in ca_histories])
            
            mean_ca = {
                'S_frac': np.mean(ca_S_values, axis=0),
                'I_frac': np.mean(ca_I_values, axis=0),
                'R_frac': np.mean(ca_R_values, axis=0),
                'S_std': np.std(ca_S_values, axis=0),
                'I_std': np.std(ca_I_values, axis=0),
                'R_std': np.std(ca_R_values, axis=0),
                'timestep': ca_histories[0]['timestep']
            }
            
            std_ca = {
                'S_frac': np.std(ca_S_values, axis=0),
                'I_frac': np.std(ca_I_values, axis=0),
                'R_frac': np.std(ca_R_values, axis=0)
            }
            
            # Calculate mean and std for ODE results
            mean_ode_states = np.mean(ode_states_list, axis=0)
            std_ode_states = np.std(ode_states_list, axis=0)
            
            # Calculate norms
            norms = calculate_norm(mean_ca, ode_time, mean_ode_states)
            norms_S_row.append(norms['S'])
            norms_I_row.append(norms['I'])
            norms_R_row.append(norms['R'])
            
            print(f"✓ S={norms['S']:.3f}, I={norms['I']:.3f}, R={norms['R']:.3f}")
            
            # Store results
            mixing_results.append((mean_ca, ode_time, mean_ode_states, std_ode_states))
            
            # Save to CSV
            save_results_to_csv(grid_idx, width, height, pair_idx, mixing_rate, initial_infected_count,
                              (mean_ca, std_ca), (ode_time, mean_ode_states, std_ode_states))
        
        grid_mixing_results.append(mixing_results)
        norms_S.append(norms_S_row)
        norms_I.append(norms_I_row)
        norms_R.append(norms_R_row)
        
        # PLOT AFTER EACH GRID SIZE COMPLETES
        print(f"\nGenerating time series plot for {width}x{height}...")
        
        # Plot time series for this grid size
        _plot_timeseries_for_grid(mixing_results, width, height, mixing_initial_infected_pairs)
        
        print(f"Time series plot saved for {width}x{height}\n")
    
    # Create labels for final heatmaps
    grid_labels = [f"{w}x{h}" for w, h in grid_sizes]
    mixing_initial_labels = [f"M:{m:.1f}\nI:{i:.0f}" for m, i in mixing_initial_infected_pairs]
    
    # Plot final complete heatmaps
    print("\n" + "="*70)
    print("Generating final complete heatmaps...")
    plot_norm_heatmaps(norms_S, norms_I, norms_R, grid_labels, mixing_initial_labels)
    
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print(f"CSV files saved to: {csv_dir}")
    print(f"Images saved to: {img_dir}")
    print("="*70)

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    grid_mixing_initial_infected_combined_sensitivity_experiment()
