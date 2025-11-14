"""
Grid Size and Mixing Rate Combined Sensitivity Analysis

Analyzes how disease dynamics (S, I, R) vary with both grid size and mixing rate.
Produces:
1. Time series comparison plots (CA vs ODE) for selected grid/mixing combinations
2. Heatmaps of L2 norms for S, I, R showing how they vary across both parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import importlib.util
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
    'initial_infected_count': 10,
    'num_simulations'       : 5
}

# Parameter ranges for combined analysis
grid_sizes = [(50, 50), (100, 100), (150, 150), (200, 200), (300, 300)]
mixing_rate_values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

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
# Time series plots for selected combinations
# ==========================================================
def plot_timeseries_for_selection(grid_mixing_results, selected_combinations):
    """
    Plot time series (CA vs ODE) for selected grid size and mixing rate combinations.
    
    selected_combinations: list of tuples [(grid_idx, mixing_idx), ...]
    """
    fig, axes = plt.subplots(len(selected_combinations), 3, figsize=(15, 4 * len(selected_combinations)))
    
    if len(selected_combinations) == 1:
        axes = axes.reshape(1, -1)
    
    for row, (grid_idx, mixing_idx) in enumerate(selected_combinations):
        width, height = grid_sizes[grid_idx]
        mixing_rate = mixing_rate_values[mixing_idx]
        
        ca_history, ode_time, ode_states = grid_mixing_results[grid_idx][mixing_idx]
        
        # Plot S, I, R
        for col, (state_idx, state_name, color) in enumerate([(0, 'S', 'blue'), (1, 'I', 'red'), (2, 'R', 'green')]):
            ax = axes[row, col]
            
            # CA plot
            ax.plot(ca_history['timestep'], [ca_history['SIRS'[state_idx] + '_frac'][i] for i in range(len(ca_history['timestep']))],
                   linestyle='--', color=color, alpha=0.6, linewidth=2, label=f'CA {state_name}')
            
            # ODE plot
            ax.plot(ode_time, ode_states[:, state_idx], linestyle='-', color=color, linewidth=2, label=f'ODE {state_name}')
            
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Fraction', fontsize=10)
            ax.set_title(f'{state_name} (Grid: {width}x{height}, Mixing: {mixing_rate:.2f})', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('timeseries_grid_mixing_combined.png', dpi=300, bbox_inches='tight')
    print("Time series plot saved to 'timeseries_grid_mixing_combined.png'")
    plt.show()

def plot_timeseries_for_selection_fixed(grid_mixing_results, selected_combinations):
    """
    Plot time series (CA vs ODE) for selected grid size and mixing rate combinations.
    Fixed version that correctly indexes the history dictionaries.
    
    selected_combinations: list of tuples [(grid_idx, mixing_idx), ...]
    """
    fig, axes = plt.subplots(len(selected_combinations), 3, figsize=(15, 4 * len(selected_combinations)))
    
    if len(selected_combinations) == 1:
        axes = axes.reshape(1, -1)
    
    for row, (grid_idx, mixing_idx) in enumerate(selected_combinations):
        width, height = grid_sizes[grid_idx]
        mixing_rate = mixing_rate_values[mixing_idx]
        
        ca_history, ode_time, ode_states = grid_mixing_results[grid_idx][mixing_idx]
        
        # Plot S, I, R
        for col, (state_idx, state_name, color) in enumerate([(0, 'S', 'blue'), (1, 'I', 'red'), (2, 'R', 'green')]):
            ax = axes[row, col]
            
            # CA plot
            ca_fracs = [ca_history['S_frac'], ca_history['I_frac'], ca_history['R_frac']]
            ax.plot(ca_history['timestep'], ca_fracs[state_idx],
                   linestyle='--', color=color, alpha=0.6, linewidth=2, label=f'CA {state_name}')
            
            # ODE plot
            ax.plot(ode_time, ode_states[:, state_idx], linestyle='-', color=color, linewidth=2, label=f'ODE {state_name}')
            
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Fraction', fontsize=10)
            ax.set_title(f'{state_name} (Grid: {width}x{height}, Mixing: {mixing_rate:.2f})', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('timeseries_grid_mixing_combined.png', dpi=300, bbox_inches='tight')
    print("Time series plot saved to 'timeseries_grid_mixing_combined.png'")
    plt.show()

# ==========================================================
# Heatmap plots
# ==========================================================
def plot_norm_heatmaps(norms_S, norms_I, norms_R, grid_labels, mixing_labels):
    """
    Plot heatmaps for L2 norms of S, I, R across grid sizes and mixing rates.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Prepare data for heatmaps
    data_S = np.array(norms_S)
    data_I = np.array(norms_I)
    data_R = np.array(norms_R)
    
    # Plot heatmap for S
    sns.heatmap(data_S, xticklabels=mixing_labels, yticklabels=grid_labels,
                cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f')
    axes[0].set_title('L2 Norm: Susceptible (S)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Mixing Rate', fontsize=11)
    axes[0].set_ylabel('Grid Size', fontsize=11)
    
    # Plot heatmap for I
    sns.heatmap(data_I, xticklabels=mixing_labels, yticklabels=grid_labels,
                cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f')
    axes[1].set_title('L2 Norm: Infected (I)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Mixing Rate', fontsize=11)
    axes[1].set_ylabel('Grid Size', fontsize=11)
    
    # Plot heatmap for R
    sns.heatmap(data_R, xticklabels=mixing_labels, yticklabels=grid_labels,
                cmap='YlOrRd', ax=axes[2], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f')
    axes[2].set_title('L2 Norm: Recovered (R)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Mixing Rate', fontsize=11)
    axes[2].set_ylabel('Grid Size', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('heatmaps_norms_grid_mixing.png', dpi=300, bbox_inches='tight')
    print("Heatmap plot saved to 'heatmaps_norms_grid_mixing.png'")
    plt.show()

# ==========================================================
# Helper plotting functions for intermediate results
# ==========================================================
def _plot_partial_heatmaps(norms_S, norms_I, norms_R, grid_labels_partial, mixing_labels, grid_idx, total_grids):
    """Plot partial heatmaps with data collected so far."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Prepare data for heatmaps
    data_S = np.array(norms_S)
    data_I = np.array(norms_I)
    data_R = np.array(norms_R)
    
    # Plot heatmap for S
    sns.heatmap(data_S, xticklabels=mixing_labels, yticklabels=grid_labels_partial,
                cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f', vmin=0, vmax=0.5)
    axes[0].set_title('L2 Norm: Susceptible (S)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Mixing Rate', fontsize=11)
    axes[0].set_ylabel('Grid Size', fontsize=11)
    
    # Plot heatmap for I
    sns.heatmap(data_I, xticklabels=mixing_labels, yticklabels=grid_labels_partial,
                cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f', vmin=0, vmax=0.5)
    axes[1].set_title('L2 Norm: Infected (I)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Mixing Rate', fontsize=11)
    axes[1].set_ylabel('Grid Size', fontsize=11)
    
    # Plot heatmap for R
    sns.heatmap(data_R, xticklabels=mixing_labels, yticklabels=grid_labels_partial,
                cmap='YlOrRd', ax=axes[2], cbar_kws={'label': 'L2 Norm'},
                annot=True, fmt='.3f', vmin=0, vmax=0.5)
    axes[2].set_title('L2 Norm: Recovered (R)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Mixing Rate', fontsize=11)
    axes[2].set_ylabel('Grid Size', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'heatmaps_partial_grid{grid_idx+1}of{total_grids}.png', dpi=150, bbox_inches='tight')
    print(f"  → Partial heatmap saved")
    plt.close()

def _plot_timeseries_for_grid(mixing_results, width, height, mixing_rates):
    """
    Plot time series for all mixing rates of a given grid size.
    Organization: 3 rows (S, I, R states) × N columns (mixing rates)
    Each cell shows CA vs ODE curves for that state and mixing rate.
    """
    num_mixing = len(mixing_rates)
    fig, axes = plt.subplots(3, num_mixing, figsize=(5 * num_mixing, 12))
    
    # Handle case with single mixing rate
    if num_mixing == 1:
        axes = axes.reshape(3, 1)
    
    # Row 0: S (Susceptible), Row 1: I (Infected), Row 2: R (Recovered)
    states = [
        (0, 'S', 'blue'),
        (1, 'I', 'red'),
        (2, 'R', 'green')
    ]
    
    for row, (state_idx, state_name, color) in enumerate(states):
        for col, (mixing_idx, mixing_rate) in enumerate(zip(range(num_mixing), mixing_rates)):
            ax = axes[row, col]
            
            ca_history, ode_time, ode_states = mixing_results[mixing_idx]
            
            # CA plot
            ca_fracs = [ca_history['S_frac'], ca_history['I_frac'], ca_history['R_frac']]
            ax.plot(ca_history['timestep'], ca_fracs[state_idx],
                   linestyle='--', color=color, alpha=0.6, linewidth=2, label=f'CA {state_name}')
            
            # ODE plot
            ax.plot(ode_time, ode_states[:, state_idx], linestyle='-', color=color, linewidth=2, label=f'ODE {state_name}')
            
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Fraction', fontsize=10)
            ax.set_title(f'{state_name} - Mixing: {mixing_rate:.2f}', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
    
    # Add a main title for the entire figure
    fig.suptitle(f'Time Series for Grid: {width}x{height} (varying mixing rate)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(f'timeseries_grid_{width}x{height}.png', dpi=150, bbox_inches='tight')
    print(f"  → Time series saved: timeseries_grid_{width}x{height}.png")
    plt.close()

# ==========================================================
# Main experiment
# ==========================================================
def grid_mixing_combined_sensitivity_experiment():
    print("="*70)
    print("Grid Size and Mixing Rate Combined Sensitivity Analysis")
    print("="*70)
    print(f"\nGrid sizes: {grid_sizes}")
    print(f"Mixing rates: {mixing_rate_values}")
    print(f"Simulations per combination: {params['num_simulations']}")
    print()
    
    # Store results: grid_mixing_results[grid_idx][mixing_idx] = (ca_history, ode_time, ode_states)
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
        
        norms_S_row = []
        norms_I_row = []
        norms_R_row = []
        
        mixing_results = []
        
        # Iterate over mixing rates
        for mixing_idx, mixing_rate in enumerate(mixing_rate_values):
            print(f"  [{mixing_idx+1}/{len(mixing_rate_values)}] Mixing rate: {mixing_rate:.2f}...", end=' ', flush=True)
            
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
                    initial_infected_count  =params['initial_infected_count'],
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
            mean_ca = {
                'S_frac': np.mean([np.array(hist['S_frac']) for hist in ca_histories], axis=0),
                'I_frac': np.mean([np.array(hist['I_frac']) for hist in ca_histories], axis=0),
                'R_frac': np.mean([np.array(hist['R_frac']) for hist in ca_histories], axis=0),
                'timestep': ca_histories[0]['timestep']
            }
            
            # Calculate mean for ODE results
            mean_ode_states = np.mean(ode_states_list, axis=0)
            
            # Calculate norms
            norms = calculate_norm(mean_ca, ode_time, mean_ode_states)
            norms_S_row.append(norms['S'])
            norms_I_row.append(norms['I'])
            norms_R_row.append(norms['R'])
            
            print(f"✓ S={norms['S']:.3f}, I={norms['I']:.3f}, R={norms['R']:.3f}")
            
            # Store results
            mixing_results.append((mean_ca, ode_time, mean_ode_states))
        
        grid_mixing_results.append(mixing_results)
        norms_S.append(norms_S_row)
        norms_I.append(norms_I_row)
        norms_R.append(norms_R_row)
        
        # PLOT AFTER EACH GRID SIZE COMPLETES
        print(f"\nGenerating plots for {width}x{height}...")
        grid_labels_partial = [f"{w}x{h}" for w, h in grid_sizes[:grid_idx+1]]
        mixing_labels = [f"{m:.1f}" for m in mixing_rate_values]
        
        # Create temporary heatmaps with data collected so far
        _plot_partial_heatmaps(norms_S, norms_I, norms_R, grid_labels_partial, 
                               mixing_labels, grid_idx, len(grid_sizes))
        
        # Plot time series for this grid size
        _plot_timeseries_for_grid(mixing_results, width, height, mixing_rate_values)
        
        print(f"Plots saved for {width}x{height}\n")
    
    # Create labels for final heatmaps
    grid_labels = [f"{w}x{h}" for w, h in grid_sizes]
    mixing_labels = [f"{m:.1f}" for m in mixing_rate_values]
    
    # Plot final complete heatmaps
    print("\n" + "="*70)
    print("Generating final complete heatmaps...")
    plot_norm_heatmaps(norms_S, norms_I, norms_R, grid_labels, mixing_labels)
    
    # # Plot time series for selected combinations
    # print("\nGenerating final time series comparison...")
    # selected_combinations = [
    #     (0, 0),  # 50x50, mixing=0.0
    #     (1, 2),  # 100x100, mixing=0.4
    #     (3, 0),  # 200x200, mixing=0.0
    #     (3, 2),  # 200x200, mixing=0.4
    # ]
    # plot_timeseries_for_selection_fixed(grid_mixing_results, selected_combinations)
    
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70)

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    grid_mixing_combined_sensitivity_experiment()
