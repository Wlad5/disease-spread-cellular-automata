"""
Multi-Parameter Sensitivity Analysis:
- Initial number of infected cells
- Mixing rate
- Grid size

This experiment explores how disease spread dynamics change when we vary
the number of initially infected cells alongside mixing rate and grid size.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
from scipy.interpolate import interp1d
import csv

# ==========================================================
# Dynamic imports of CA and ODE modules
# ==========================================================
ca_path = os.path.join(os.path.dirname(__file__), "SIRS.py")
ode_path = os.path.join(os.path.dirname(__file__), "SIRS_ODE_SOLVER.py")

def import_module_from_path(module_name, file_path):
    spec    = importlib.util.spec_from_file_location(module_name, file_path)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

SIRS_CA = import_module_from_path("SIRS_CA", ca_path)
SIRS_ODE = import_module_from_path("SIRS_ODE", ode_path)

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
    'num_simulations'       : 3  # Number of simulations per parameter value
}

# Parameter ranges for multi-dimensional sensitivity analysis
grid_sizes = [25, 50, 100]                          # Grid size (N x N)
mixing_rates = [0.0, 0.05, 0.1]                     # Mixing rate
initial_infected_fractions = [0.01, 0.05, 0.1]     # Initial infected as fraction of grid

# ==========================================================
# Helper: compute initial infected count from fraction
# ==========================================================
def compute_initial_infected_count(width, height, infected_fraction):
    """Compute the number of initially infected cells from a fraction."""
    total_cells = width * height
    count = max(1, int(infected_fraction * total_cells))
    actual_fraction = count / total_cells
    return count, actual_fraction

# ==========================================================
# Simulation runners
# ==========================================================
def run_ca_simulation(infection_prob, recovery_prob, waning_prob, t_max,
                      width, height, cell_size, initial_infected_count, mixing_rate=0.05):
    """Run CA simulation and return history + initial infected fraction."""
    sim = SIRS_CA.Simulation(
        width=width,
        height=height,
        cell_size=cell_size,
        infection_prob=infection_prob,
        recovery_prob=recovery_prob,
        waning_prob=waning_prob,
        delta_t=1,
        mixing_rate=mixing_rate
    )

    # Infect a given number of random cells
    sim.grid.infect_random(n=initial_infected_count)

    # Compute actual initial infected fraction
    total_cells = width * height
    initial_infected_frac = initial_infected_count / total_cells

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
        infection_prob=infection_prob,
        recovery_prob=recovery_prob,
        waning_prob=waning_prob,
        k=k,
        delta_t=delta_t,
        initial_infected=initial_infected_frac,
        t_max=t_max,
        dt=ode_dt
    )
    return time_points, states

# ==========================================================
# Plotting functions
# ==========================================================
def plot_peak_infected_heatmap(grid_sizes, mixing_rates, initial_infected_fractions, 
                               peak_infected_data, title="Peak Infected Count"):
    """
    Create a heatmap showing peak infected across parameters.
    Data structure: peak_infected_data[grid_idx][mixing_idx][init_frac_idx]
    """
    fig, axes = plt.subplots(1, len(grid_sizes), figsize=(15, 4))
    
    for grid_idx, grid_size in enumerate(grid_sizes):
        data = np.array([[peak_infected_data[grid_idx][mixing_idx][init_frac_idx] 
                         for init_frac_idx in range(len(initial_infected_fractions))]
                         for mixing_idx in range(len(mixing_rates))])
        
        im = axes[grid_idx].imshow(data, cmap='viridis', aspect='auto')
        axes[grid_idx].set_xlabel('Initial Infected Fraction')
        axes[grid_idx].set_ylabel('Mixing Rate')
        axes[grid_idx].set_title(f'Grid {grid_size}x{grid_size}')
        axes[grid_idx].set_xticks(range(len(initial_infected_fractions)))
        axes[grid_idx].set_xticklabels([f'{f:.2f}' for f in initial_infected_fractions])
        axes[grid_idx].set_yticks(range(len(mixing_rates)))
        axes[grid_idx].set_yticklabels([f'{m:.2f}' for m in mixing_rates])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[grid_idx])
        cbar.set_label('Peak Infected')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_trajectories_grid(grid_sizes, mixing_rates, initial_infected_fractions,
                          ca_results, ode_results):
    """
    Plot trajectories for a subset of parameter combinations.
    """
    # Select a few representative combinations to plot
    fig, axes = plt.subplots(len(mixing_rates), len(grid_sizes), figsize=(14, 10))
    
    for grid_idx, grid_size in enumerate(grid_sizes):
        for mixing_idx, mixing_rate in enumerate(mixing_rates):
            ax = axes[mixing_idx, grid_idx]
            
            # Use first initial infected fraction for trajectory plot
            init_frac_idx = 0
            
            mean_ca, std_ca = ca_results[grid_idx][mixing_idx][init_frac_idx]
            ode_time, mean_ode_states, std_ode_states = ode_results[grid_idx][mixing_idx][init_frac_idx]
            
            # Plot CA results
            ax.plot(mean_ca['timestep'], mean_ca['I_frac'], 'r--', linewidth=2, label='CA I')
            ax.fill_between(mean_ca['timestep'], 
                            mean_ca['I_frac'] - std_ca['I_frac'], 
                            mean_ca['I_frac'] + std_ca['I_frac'], 
                            color='r', alpha=0.2)
            
            # Plot ODE results
            ax.plot(ode_time, mean_ode_states[:, 1], 'r-', linewidth=2, label='ODE I')
            ax.fill_between(ode_time,
                            mean_ode_states[:, 1] - std_ode_states[:, 1],
                            mean_ode_states[:, 1] + std_ode_states[:, 1],
                            color='r', alpha=0.2)
            
            ax.set_title(f'Grid {grid_size}x{grid_size}, Mix={mixing_rate:.2f}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Infected Fraction')
            ax.set_ylim(0, 1)
            if mixing_idx == 0 and grid_idx == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Infected Trajectories: CA vs ODE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==========================================================
# Main experiment
# ==========================================================
def multi_parameter_sensitivity_experiment():
    print("="*70)
    print("Multi-Parameter Sensitivity Analysis")
    print("Parameters: Grid Size, Mixing Rate, Initial Infected Count")
    print("="*70)
    
    # Storage for results
    ca_results = {}      # [grid_idx][mixing_idx][init_frac_idx] = (mean_ca, std_ca)
    ode_results = {}     # [grid_idx][mixing_idx][init_frac_idx] = (ode_time, mean_ode_states, std_ode_states)
    peak_infected_ca = {}  # [grid_idx][mixing_idx][init_frac_idx] = peak value
    
    total_combinations = len(grid_sizes) * len(mixing_rates) * len(initial_infected_fractions)
    completed = 0
    
    for grid_idx, grid_size in enumerate(grid_sizes):
        ca_results[grid_idx] = {}
        ode_results[grid_idx] = {}
        peak_infected_ca[grid_idx] = {}
        
        for mixing_idx, mixing_rate in enumerate(mixing_rates):
            ca_results[grid_idx][mixing_idx] = {}
            ode_results[grid_idx][mixing_idx] = {}
            peak_infected_ca[grid_idx][mixing_idx] = {}
            
            for init_frac_idx, init_frac in enumerate(initial_infected_fractions):
                # Compute actual initial infected count
                init_count, actual_frac = compute_initial_infected_count(
                    grid_size, grid_size, init_frac
                )
                
                print(f"\n[{completed+1}/{total_combinations}] Running simulations:")
                print(f"  Grid: {grid_size}x{grid_size} ({grid_size**2} cells)")
                print(f"  Mixing Rate: {mixing_rate:.3f}")
                print(f"  Initial Infected: {init_count} cells ({actual_frac:.4f})")
                
                ca_histories = []
                ode_states_list = []
                
                # Run multiple simulations for averaging
                for sim_num in range(params['num_simulations']):
                    ca_hist, ca_initial_infected_frac = run_ca_simulation(
                        infection_prob=params['infection_prob'],
                        recovery_prob=params['recovery_prob'],
                        waning_prob=params['waning_prob'],
                        t_max=params['t_max'],
                        width=grid_size,
                        height=grid_size,
                        cell_size=4,
                        initial_infected_count=init_count,
                        mixing_rate=mixing_rate
                    )
                    
                    ode_time, ode_states = run_ode_simulation(
                        infection_prob=params['infection_prob'],
                        recovery_prob=params['recovery_prob'],
                        waning_prob=params['waning_prob'],
                        k=params['k'],
                        delta_t=params['delta_t'],
                        initial_infected_frac=ca_initial_infected_frac,
                        t_max=params['t_max'],
                        ode_dt=params['ode_dt']
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
                std_ca = {
                    'S_frac': np.std([np.array(hist['S_frac']) for hist in ca_histories], axis=0),
                    'I_frac': np.std([np.array(hist['I_frac']) for hist in ca_histories], axis=0),
                    'R_frac': np.std([np.array(hist['R_frac']) for hist in ca_histories], axis=0),
                }
                
                # Calculate mean and std for ODE results
                mean_ode_states = np.mean(ode_states_list, axis=0)
                std_ode_states = np.std(ode_states_list, axis=0)
                
                # Store results
                ca_results[grid_idx][mixing_idx][init_frac_idx] = (mean_ca, std_ca)
                ode_results[grid_idx][mixing_idx][init_frac_idx] = (ode_time, mean_ode_states, std_ode_states)
                
                # Calculate peak infected
                peak_infected_ca[grid_idx][mixing_idx][init_frac_idx] = np.max(mean_ca['I_frac'])
                
                print(f"  Peak Infected (CA): {peak_infected_ca[grid_idx][mixing_idx][init_frac_idx]:.4f}")
                
                completed += 1
    
    print("\n" + "="*70)
    print("All simulations completed. Generating plots...")
    print("="*70)
    
    # Generate heatmaps
    plot_peak_infected_heatmap(grid_sizes, mixing_rates, initial_infected_fractions,
                              peak_infected_ca, "Peak Infected Across Parameters")
    
    # Generate trajectory plots
    plot_trajectories_grid(grid_sizes, mixing_rates, initial_infected_fractions,
                          ca_results, ode_results)
    
    # Save summary to CSV
    save_results_summary(grid_sizes, mixing_rates, initial_infected_fractions, peak_infected_ca)

def save_results_summary(grid_sizes, mixing_rates, initial_infected_fractions, peak_infected_data):
    """Save summary of results to CSV file."""
    filename = "initial_infected_mixing_grid_sensitivity_results.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Grid_Size', 'Mixing_Rate', 'Initial_Infected_Fraction', 'Peak_Infected'])
        
        for grid_idx, grid_size in enumerate(grid_sizes):
            for mixing_idx, mixing_rate in enumerate(mixing_rates):
                for init_frac_idx, init_frac in enumerate(initial_infected_fractions):
                    peak_val = peak_infected_data[grid_idx][mixing_idx][init_frac_idx]
                    writer.writerow([grid_size, mixing_rate, init_frac, peak_val])
    
    print(f"Results saved to {filename}")

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    multi_parameter_sensitivity_experiment()
