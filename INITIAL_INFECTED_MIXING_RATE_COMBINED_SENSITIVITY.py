import numpy as np
import matplotlib.pyplot as plt
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
    'width'                 : 50,
    'height'                : 50,
    'cell_size'             : 4,
    'initial_infected_count': 10,
    'mixing_rate'           : 0.00,
    'num_simulations'       : 5  # Number of simulations per parameter combination
}

# Combined parameter ranges for sensitivity analysis
initial_infected_counts = [5, 50, 500, 1000]  # Number of initially infected cells
mixing_rate_values = np.linspace(0.0, 1.0, 4)  # Mixing rates from 0 to 0.5

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
# Plotting with Error Bars - Grid Layout
# ==========================================================
def plot_comparison_with_error_grid(initial_infected_counts, mixing_rate_values, 
                                     ca_results, ode_results, t_max, ode_dt):
    """
    Create a grid of subplots where:
    - Rows correspond to initial infected counts
    - Columns correspond to mixing rates
    - Each cell shows CA vs ODE comparison for that parameter pair
    """
    n_infected = len(initial_infected_counts)
    n_mixing = len(mixing_rate_values)
    
    fig, axes = plt.subplots(n_infected, n_mixing, figsize=(16, 12))
    
    for i, infected_count in enumerate(initial_infected_counts):
        for j, mixing_rate in enumerate(mixing_rate_values):
            ax = axes[i, j]
            
            # Get results for this parameter combination
            result_idx = i * n_mixing + j
            mean_ca, std_ca = ca_results[result_idx]
            ode_time, mean_ode_states, std_ode_states = ode_results[result_idx]
            
            # Plot CA results
            ax.plot(mean_ca['timestep'], mean_ca['S_frac'], label='CA S', color='b', linestyle='--', alpha=0.7)
            ax.fill_between(mean_ca['timestep'], 
                           mean_ca['S_frac'] - std_ca['S_frac'], 
                           mean_ca['S_frac'] + std_ca['S_frac'], 
                           color='b', alpha=0.1)
            ax.plot(mean_ca['timestep'], mean_ca['I_frac'], label='CA I', color='r', linestyle='--', alpha=0.7)
            ax.fill_between(mean_ca['timestep'], 
                           mean_ca['I_frac'] - std_ca['I_frac'], 
                           mean_ca['I_frac'] + std_ca['I_frac'], 
                           color='r', alpha=0.1)
            ax.plot(mean_ca['timestep'], mean_ca['R_frac'], label='CA R', color='g', linestyle='--', alpha=0.7)
            ax.fill_between(mean_ca['timestep'], 
                           mean_ca['R_frac'] - std_ca['R_frac'], 
                           mean_ca['R_frac'] + std_ca['R_frac'], 
                           color='g', alpha=0.1)
            
            # Plot ODE results
            ax.plot(ode_time, mean_ode_states[:, 0], label='ODE S', color='b', linestyle='-', linewidth=2)
            ax.fill_between(ode_time, 
                           mean_ode_states[:, 0] - std_ode_states[:, 0], 
                           mean_ode_states[:, 0] + std_ode_states[:, 0], 
                           color='b', alpha=0.15)
            ax.plot(ode_time, mean_ode_states[:, 1], label='ODE I', color='r', linestyle='-', linewidth=2)
            ax.fill_between(ode_time, 
                           mean_ode_states[:, 1] - std_ode_states[:, 1], 
                           mean_ode_states[:, 1] + std_ode_states[:, 1], 
                           color='r', alpha=0.15)
            ax.plot(ode_time, mean_ode_states[:, 2], label='ODE R', color='g', linestyle='-', linewidth=2)
            ax.fill_between(ode_time, 
                           mean_ode_states[:, 2] - std_ode_states[:, 2], 
                           mean_ode_states[:, 2] + std_ode_states[:, 2], 
                           color='g', alpha=0.15)
            
            # Labels and formatting
            ax.set_title(f"I₀={infected_count}, mixing={mixing_rate:.2f}", fontsize=11, fontweight='bold')
            ax.set_xlabel('Time', fontsize=9)
            ax.set_ylabel('Fraction', fontsize=9)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8, ncol=3)
    
    plt.suptitle("Combined Sensitivity: Initial Infected Count × Mixing Rate", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

# ==========================================================
# Norm Calculation and Heatmap Plotting
# ==========================================================
def calculate_norms_grid(ca_results, ode_results, initial_infected_counts, mixing_rate_values):
    """
    Calculate L2 norms for each parameter combination.
    
    Returns:
        norms_dict: Dictionary with 'S', 'I', 'R' keys, each containing
                   2D arrays of norms (rows: initial_infected, cols: mixing_rate)
    """
    n_infected = len(initial_infected_counts)
    n_mixing = len(mixing_rate_values)
    
    norms_S = np.zeros((n_infected, n_mixing))
    norms_I = np.zeros((n_infected, n_mixing))
    norms_R = np.zeros((n_infected, n_mixing))
    
    for i, infected_count in enumerate(initial_infected_counts):
        for j, mixing_rate in enumerate(mixing_rate_values):
            result_idx = i * n_mixing + j
            mean_ca, std_ca = ca_results[result_idx]
            ode_time, mean_ode_states, std_ode_states = ode_results[result_idx]
            
            # Interpolate ODE results to match CA time steps
            ca_timesteps = np.array(mean_ca['timestep'])
            ode_time_interp = np.linspace(0, len(mean_ode_states) * params['ode_dt'], len(mean_ode_states))
            interp_ode_S = interp1d(ode_time_interp, mean_ode_states[:, 0], kind='linear', fill_value="extrapolate")
            interp_ode_I = interp1d(ode_time_interp, mean_ode_states[:, 1], kind='linear', fill_value="extrapolate")
            interp_ode_R = interp1d(ode_time_interp, mean_ode_states[:, 2], kind='linear', fill_value="extrapolate")
            
            # Calculate L2 norms
            norms_S[i, j] = np.sqrt(np.sum((np.array(mean_ca['S_frac']) - interp_ode_S(ca_timesteps)) ** 2))
            norms_I[i, j] = np.sqrt(np.sum((np.array(mean_ca['I_frac']) - interp_ode_I(ca_timesteps)) ** 2))
            norms_R[i, j] = np.sqrt(np.sum((np.array(mean_ca['R_frac']) - interp_ode_R(ca_timesteps)) ** 2))
    
    return {'S': norms_S, 'I': norms_I, 'R': norms_R}

def plot_norms_heatmap(initial_infected_counts, mixing_rate_values, norms_dict):
    """Plot norms as heatmaps for each compartment."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Labels for axes
    x_labels = [f"{m:.2f}" for m in mixing_rate_values]
    y_labels = [str(c) for c in initial_infected_counts]
    
    for idx, (compartment, ax) in enumerate(zip(['S', 'I', 'R'], axes)):
        norms = norms_dict[compartment]
        im = ax.imshow(norms, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(mixing_rate_values)))
        ax.set_yticks(np.arange(len(initial_infected_counts)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(initial_infected_counts)):
            for j in range(len(mixing_rate_values)):
                text = ax.text(j, i, f'{norms[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(f'L2 Norm - Compartment {compartment}', fontweight='bold')
        ax.set_xlabel('Mixing Rate', fontsize=10)
        ax.set_ylabel('Initial Infected Count', fontsize=10)
        fig.colorbar(im, ax=ax)
    
    plt.suptitle('CA vs ODE Norms - Grid Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==========================================================
# Main experiment
# ==========================================================
def combined_sensitivity_experiment():
    total_cells = params['width'] * params['height']
    print(f"Grid: {params['width']}x{params['height']} ({total_cells} cells)")
    print(f"\nRunning combined sensitivity analysis...")
    print(f"Initial infected counts: {initial_infected_counts}")
    print(f"Mixing rates: {mixing_rate_values}\n")

    ca_results = []
    ode_results = []
    result_count = 0

    for infected_count in initial_infected_counts:
        for mixing_rate in mixing_rate_values:
            print(f"Running: I₀={infected_count}, mixing_rate={mixing_rate:.3f}")
            
            p = params.copy()
            p['initial_infected_count'] = infected_count
            p['mixing_rate'] = mixing_rate

            ca_histories = []
            ode_states_list = []

            for sim_idx in range(params['num_simulations']):
                ca_hist, ca_initial_infected_frac = run_ca_simulation(
                    infection_prob          =p['infection_prob'],
                    recovery_prob           =p['recovery_prob'],
                    waning_prob             =p['waning_prob'],
                    t_max                   =p['t_max'],
                    width                   =p['width'],
                    height                  =p['height'],
                    cell_size               =p['cell_size'],
                    initial_infected_count  =p['initial_infected_count'],
                    mixing_rate             =p['mixing_rate']
                )
                ode_time, ode_states = run_ode_simulation(
                    infection_prob          =p['infection_prob'],
                    recovery_prob           =p['recovery_prob'],
                    waning_prob             =p['waning_prob'],
                    k                       =p['k'],
                    delta_t                 =p['delta_t'],
                    initial_infected_frac   =ca_initial_infected_frac,
                    t_max                   =p['t_max'],
                    ode_dt                  =p['ode_dt']
                )

                ca_histories.append(ca_hist)
                ode_states_list.append(ode_states)

            # Calculate mean and std for CA results
            ca_fracs_S = np.array([np.array(hist['S_frac']) for hist in ca_histories])
            ca_fracs_I = np.array([np.array(hist['I_frac']) for hist in ca_histories])
            ca_fracs_R = np.array([np.array(hist['R_frac']) for hist in ca_histories])
            
            mean_ca = {
                'S_frac': np.mean(ca_fracs_S, axis=0),
                'I_frac': np.mean(ca_fracs_I, axis=0),
                'R_frac': np.mean(ca_fracs_R, axis=0),
                'timestep': ca_histories[0]['timestep']
            }
            std_ca = {
                'S_frac': np.std(ca_fracs_S, axis=0),
                'I_frac': np.std(ca_fracs_I, axis=0),
                'R_frac': np.std(ca_fracs_R, axis=0),
            }

            # Calculate mean and std for ODE results
            mean_ode_states = np.mean(ode_states_list, axis=0)
            std_ode_states  = np.std(ode_states_list, axis=0)

            print(f"  ✓ Completed (I₀ frac: {ca_initial_infected_frac:.5f})")

            ca_results.append((mean_ca, std_ca))
            ode_results.append((ode_time, mean_ode_states, std_ode_states))
            result_count += 1

    print(f"\nCompleted {result_count} parameter combinations with {params['num_simulations']} simulations each")
    print("Generating plots...\n")

    # Plot comparison grid
    plot_comparison_with_error_grid(initial_infected_counts, mixing_rate_values, 
                                    ca_results, ode_results, params['t_max'], params['ode_dt'])

    # Calculate and plot norms
    norms = calculate_norms_grid(ca_results, ode_results, initial_infected_counts, mixing_rate_values)
    plot_norms_heatmap(initial_infected_counts, mixing_rate_values, norms)

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    combined_sensitivity_experiment()
