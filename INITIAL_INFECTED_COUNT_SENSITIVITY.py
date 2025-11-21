import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
import csv
from scipy.interpolate import interp1d

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
    't_max'                 : 100,
    'dt'                    : 1.0,
    'ode_dt'                : 0.1,
    'width'                 : 100,
    'height'                : 100,
    'cell_size'             : 4,
    'mixing_rate'           : 0.00,
    'num_simulations'       : 5
}

# Parameter range for initial infected count sensitivity analysis
initial_infected_counts = np.array([5, 20, 50, 100])
initial_infected_counts = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 7000, 10000])
plots_per_figure = 5  # Show 4 plots per figure

# ==========================================================
# Create output directories
# ==========================================================
def create_output_directories():
    """Create directories for saving CSV files and images."""
    csv_dir = os.path.join(os.path.dirname(__file__), "initial_infected_count_csv")
    img_dir = os.path.join(os.path.dirname(__file__), "initial_infected_count_images")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    return csv_dir, img_dir

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
                      width, height, cell_size, initial_infected_count, mixing_rate=0.00):
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

    # Clear any default infections and infect the specified number of random cells
    for y in range(height):
        for x in range(width):
            sim.grid.grid[y][x].state = SIRS_CA.SUSCEPTIBLE
    sim.grid.infect_random(n=initial_infected_count)

    # Compute actual initial infected fraction (used by ODE)
    initial_infected_frac = compute_initial_infected_fraction(width, height, initial_infected_count)

    # Run CA simulation steps (with visual display)
    import pygame
    for step in range(t_max):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                import sys
                sys.exit()
        sim.grid.update()
        sim.draw()
        sim.clock.tick(60)

    history = sim.get_history()
    pygame.quit()
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
# CSV saving functions
# ==========================================================
def save_ca_results_to_csv(csv_dir, initial_infected_count, mean_ca, std_ca):
    """Save CA results to CSV file."""
    filename = os.path.join(csv_dir, f"ca_initial_infected_{initial_infected_count}.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestep', 'S_frac_mean', 'S_frac_std', 'I_frac_mean', 'I_frac_std', 'R_frac_mean', 'R_frac_std'])
        for i in range(len(mean_ca['timestep'])):
            writer.writerow([
                mean_ca['timestep'][i],
                mean_ca['S_frac'][i],
                std_ca['S_frac'][i],
                mean_ca['I_frac'][i],
                std_ca['I_frac'][i],
                mean_ca['R_frac'][i],
                std_ca['R_frac'][i]
            ])
    print(f"  Saved CA results to {filename}")

def save_ode_results_to_csv(csv_dir, initial_infected_count, ode_time, mean_ode_states, std_ode_states):
    """Save ODE results to CSV file."""
    filename = os.path.join(csv_dir, f"ode_initial_infected_{initial_infected_count}.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'S_mean', 'S_std', 'I_mean', 'I_std', 'R_mean', 'R_std'])
        for i in range(len(ode_time)):
            writer.writerow([
                ode_time[i],
                mean_ode_states[i, 0],
                std_ode_states[i, 0],
                mean_ode_states[i, 1],
                std_ode_states[i, 1],
                mean_ode_states[i, 2],
                std_ode_states[i, 2]
            ])
    print(f"  Saved ODE results to {filename}")

# ==========================================================
# Plotting with Error Bars (in batches)
# ==========================================================
def plot_comparison_with_error(param_name, param_values, ca_results, ode_results, t_max, ode_dt, img_dir, plots_per_figure=4):
    num_params = len(param_values)
    num_figures = int(np.ceil(num_params / plots_per_figure))
    
    for fig_idx in range(num_figures):
        plt.figure(figsize=(18, 14))
        start_idx = fig_idx * plots_per_figure
        end_idx = min(start_idx + plots_per_figure, num_params)
        batch_size = end_idx - start_idx
        
        for batch_pos, i in enumerate(range(start_idx, end_idx)):
            val = param_values[i]
            mean_ca, std_ca = ca_results[i]
            ode_time, mean_ode_states, std_ode_states = ode_results[i]

            # ---- Row 1: S (Susceptible) ----
            ax_s = plt.subplot(3, batch_size, batch_pos + 1)
            ax_s.plot(mean_ca['timestep'], mean_ca['S_frac'], label='CA S', color='b', linestyle='--', linewidth=2)
            ax_s.fill_between(mean_ca['timestep'], mean_ca['S_frac'] - std_ca['S_frac'], mean_ca['S_frac'] + std_ca['S_frac'], color='b', alpha=0.2)
            ax_s.plot(ode_time, mean_ode_states[:, 0], label='ODE S', color='b', linewidth=2)
            ax_s.fill_between(ode_time, mean_ode_states[:, 0] - std_ode_states[:, 0], mean_ode_states[:, 0] + std_ode_states[:, 0], color='b', alpha=0.1)
            ax_s.set_title(f"S (Susceptible): {param_name}={int(val)}", fontsize=11, fontweight='bold')
            ax_s.set_ylabel('Fraction', fontsize=10)
            ax_s.set_ylim(0, 1)
            ax_s.grid(True, alpha=0.3)
            if batch_pos == 0:
                ax_s.legend(loc='best', fontsize=9)
            if batch_pos > 0:
                ax_s.set_yticklabels([])

            # ---- Row 2: I (Infected) ----
            ax_i = plt.subplot(3, batch_size, batch_size + batch_pos + 1)
            ax_i.plot(mean_ca['timestep'], mean_ca['I_frac'], label='CA I', color='r', linestyle='--', linewidth=2)
            ax_i.fill_between(mean_ca['timestep'], mean_ca['I_frac'] - std_ca['I_frac'], mean_ca['I_frac'] + std_ca['I_frac'], color='r', alpha=0.2)
            ax_i.plot(ode_time, mean_ode_states[:, 1], label='ODE I', color='r', linewidth=2)
            ax_i.fill_between(ode_time, mean_ode_states[:, 1] - std_ode_states[:, 1], mean_ode_states[:, 1] + std_ode_states[:, 1], color='r', alpha=0.1)
            ax_i.set_title(f"I (Infected): {param_name}={int(val)}", fontsize=11, fontweight='bold')
            ax_i.set_ylabel('Fraction', fontsize=10)
            ax_i.set_ylim(0, 1)
            ax_i.grid(True, alpha=0.3)
            if batch_pos == 0:
                ax_i.legend(loc='best', fontsize=9)
            if batch_pos > 0:
                ax_i.set_yticklabels([])

            # ---- Row 3: R (Recovered) ----
            ax_r = plt.subplot(3, batch_size, 2 * batch_size + batch_pos + 1)
            ax_r.plot(mean_ca['timestep'], mean_ca['R_frac'], label='CA R', color='g', linestyle='--', linewidth=2)
            ax_r.fill_between(mean_ca['timestep'], mean_ca['R_frac'] - std_ca['R_frac'], mean_ca['R_frac'] + std_ca['R_frac'], color='g', alpha=0.2)
            ax_r.plot(ode_time, mean_ode_states[:, 2], label='ODE R', color='g', linewidth=2)
            ax_r.fill_between(ode_time, mean_ode_states[:, 2] - std_ode_states[:, 2], mean_ode_states[:, 2] + std_ode_states[:, 2], color='g', alpha=0.1)
            ax_r.set_title(f"R (Recovered): {param_name}={int(val)}", fontsize=11, fontweight='bold')
            ax_r.set_xlabel('Time', fontsize=10)
            ax_r.set_ylabel('Fraction', fontsize=10)
            ax_r.set_ylim(0, 1)
            ax_r.grid(True, alpha=0.3)
            if batch_pos == 0:
                ax_r.legend(loc='best', fontsize=9)
            if batch_pos > 0:
                ax_r.set_yticklabels([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"Initial Infected Count Sensitivity (Batch {fig_idx + 1}/{num_figures})", fontsize=16, fontweight='bold', y=0.98)
        plt.subplots_adjust(hspace=0.35, wspace=0.15)
        
        # Save figure
        fig_filename = os.path.join(img_dir, f"initial_infected_sensitivity_batch_{fig_idx + 1}.png")
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        print(f"  Saved figure to {fig_filename}")
        plt.close()

# ==========================================================
# Norm Calculation
# ==========================================================
def calculate_norm(ca_results, ode_results):
    """Calculate L2 norms between CA and ODE results."""
    norms = {'S': [], 'I': [], 'R': []}
    for i in range(len(ca_results)):
        mean_ca, _ = ca_results[i]
        ode_time, mean_ode_states, _ = ode_results[i]

        # Interpolate ODE results to match CA time steps
        ca_timesteps = np.array(mean_ca['timestep'])
        ode_states_s = interp1d(ode_time, mean_ode_states[:, 0], kind='linear', fill_value="extrapolate")
        ode_states_i = interp1d(ode_time, mean_ode_states[:, 1], kind='linear', fill_value="extrapolate")
        ode_states_r = interp1d(ode_time, mean_ode_states[:, 2], kind='linear', fill_value="extrapolate")

        # Calculate L2 norms for S, I, and R
        ca_s = np.array(mean_ca['S_frac'])
        ca_i = np.array(mean_ca['I_frac'])
        ca_r = np.array(mean_ca['R_frac'])
        
        interp_ode_s = ode_states_s(ca_timesteps)
        interp_ode_i = ode_states_i(ca_timesteps)
        interp_ode_r = ode_states_r(ca_timesteps)

        norm_S = np.sqrt(np.sum((ca_s - interp_ode_s) ** 2))
        norm_I = np.sqrt(np.sum((ca_i - interp_ode_i) ** 2))
        norm_R = np.sqrt(np.sum((ca_r - interp_ode_r) ** 2))

        norms['S'].append(norm_S)
        norms['I'].append(norm_I)
        norms['R'].append(norm_R)

    return norms

def plot_norm_vs_parameter(param_name, param_values, norms, img_dir):
    """Plot L2 norms as a function of initial infected count."""
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, norms['S'], marker='o', linestyle='-', linewidth=2.5, markersize=8, label='Norm (S)', color='b')
    plt.plot(param_values, norms['I'], marker='s', linestyle='-', linewidth=2.5, markersize=8, label='Norm (I)', color='r')
    plt.plot(param_values, norms['R'], marker='^', linestyle='-', linewidth=2.5, markersize=8, label='Norm (R)', color='g')
    plt.xlabel(f'{param_name}', fontsize=12, fontweight='bold')
    plt.ylabel('L2 Norm', fontsize=12, fontweight='bold')
    plt.title(f'L2 Norms vs {param_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    fig_filename = os.path.join(img_dir, "l2_norm_vs_initial_infected.png")
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"  Saved L2 norm plot to {fig_filename}")
    plt.close()

def save_norms_to_csv(csv_dir, param_values, norms):
    """Save L2 norms to CSV file."""
    filename = os.path.join(csv_dir, "l2_norms.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['initial_infected_count', 'norm_S', 'norm_I', 'norm_R'])
        for i in range(len(param_values)):
            writer.writerow([
                int(param_values[i]),
                norms['S'][i],
                norms['I'][i],
                norms['R'][i]
            ])
    print(f"  Saved L2 norms to {filename}")

# ==========================================================
# Main experiment
# ==========================================================
def initial_infected_count_sensitivity_experiment():
    csv_dir, img_dir = create_output_directories()
    
    total_cells = params['width'] * params['height']
    print(f"Grid: {params['width']}x{params['height']} ({total_cells} cells)")
    print(f"Parameters: infection_prob={params['infection_prob']}, recovery_prob={params['recovery_prob']}, waning_prob={params['waning_prob']}")
    print(f"Output directories:")
    print(f"  CSV: {csv_dir}")
    print(f"  Images: {img_dir}\n")

    print(f"Running sensitivity analysis for initial infected count...")
    ca_results = []
    ode_results = []

    for count in initial_infected_counts:
        print(f"\n  Processing initial_infected_count={int(count)}...")
        p = params.copy()
        p['initial_infected_count'] = int(count)

        ca_histories = []
        ode_states_list = []

        for sim_idx in range(params['num_simulations']):
            ca_hist, ca_initial_infected_frac = run_ca_simulation(
                infection_prob=p['infection_prob'],
                recovery_prob=p['recovery_prob'],
                waning_prob=p['waning_prob'],
                t_max=p['t_max'],
                width=p['width'],
                height=p['height'],
                cell_size=p['cell_size'],
                initial_infected_count=p['initial_infected_count'],
                mixing_rate=p['mixing_rate']
            )
            ode_time, ode_states = run_ode_simulation(
                infection_prob=p['infection_prob'],
                recovery_prob=p['recovery_prob'],
                waning_prob=p['waning_prob'],
                k=p['k'],
                delta_t=p['delta_t'],
                initial_infected_frac=ca_initial_infected_frac,
                t_max=p['t_max'],
                ode_dt=p['ode_dt']
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

        # Save CSV results
        save_ca_results_to_csv(csv_dir, int(count), mean_ca, std_ca)
        save_ode_results_to_csv(csv_dir, int(count), ode_time, mean_ode_states, std_ode_states)

        ca_results.append((mean_ca, std_ca))
        ode_results.append((ode_time, mean_ode_states, std_ode_states))

    print(f"\n\nGenerating comparison plots...")
    plot_comparison_with_error("Initial Infected Count", initial_infected_counts, ca_results, ode_results, params['t_max'], params['ode_dt'], img_dir, plots_per_figure=plots_per_figure)

    print(f"\nCalculating L2 norms...")
    norms = calculate_norm(ca_results, ode_results)
    
    print(f"Plotting L2 norms...")
    plot_norm_vs_parameter("Initial Infected Count", initial_infected_counts, norms, img_dir)
    
    print(f"Saving L2 norms to CSV...")
    save_norms_to_csv(csv_dir, initial_infected_counts, norms)

    print(f"\n{'='*70}")
    print(f"Analysis complete!")
    print(f"Results saved to:")
    print(f"  CSV files: {csv_dir}")
    print(f"  Images: {img_dir}")
    print(f"{'='*70}")

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    initial_infected_count_sensitivity_experiment()
