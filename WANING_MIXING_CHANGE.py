import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
import csv
from pathlib import Path

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
    'width'                 : 50,
    'height'                : 50,
    'cell_size'             : 4,
    'initial_infected_count': 100,
    'mixing_rate'           : 0.00,
    'num_simulations'       : 2  # Number of simulations per parameter value
}

# Parameter ranges for waning and mixing rates (change together)
waning_prob_values = np.linspace(0.0, 1.0, 5)
mixing_rate_values = np.linspace(0.0, 1.0, 5)
waning_mixing_pairs = list(zip(waning_prob_values, mixing_rate_values))

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
# Plotting with Error Bars (simplified)
# ==========================================================
def plot_ca_ode_comparison(waning_val, mixing_val, mean_ca, std_ca, ode_time, mean_ode_states, std_ode_states, t_max, initial_infected_count, csv_dir, img_dir):
    """Plot CA vs ODE for susceptible, infected, and recovered with error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # ---- Susceptible (S) ----
    ax = axes[0]
    ax.plot(mean_ca['timestep'], mean_ca['S_frac'], label='CA S', color='b', linestyle='--', linewidth=2)
    ax.fill_between(mean_ca['timestep'], mean_ca['S_frac'] - std_ca['S_frac'], mean_ca['S_frac'] + std_ca['S_frac'], color='b', alpha=0.2)
    ax.plot(ode_time, mean_ode_states[:, 0], label='ODE S', color='b', linewidth=2)
    ax.fill_between(ode_time, mean_ode_states[:, 0] - std_ode_states[:, 0], mean_ode_states[:, 0] + std_ode_states[:, 0], color='b', alpha=0.1)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Susceptible Fraction', fontsize=12)
    ax.set_title('Susceptible (S)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # ---- Infected (I) ----
    ax = axes[1]
    ax.plot(mean_ca['timestep'], mean_ca['I_frac'], label='CA I', color='r', linestyle='--', linewidth=2)
    ax.fill_between(mean_ca['timestep'], mean_ca['I_frac'] - std_ca['I_frac'], mean_ca['I_frac'] + std_ca['I_frac'], color='r', alpha=0.2)
    ax.plot(ode_time, mean_ode_states[:, 1], label='ODE I', color='r', linewidth=2)
    ax.fill_between(ode_time, mean_ode_states[:, 1] - std_ode_states[:, 1], mean_ode_states[:, 1] + std_ode_states[:, 1], color='r', alpha=0.1)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Infected Fraction', fontsize=12)
    ax.set_title('Infected (I)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # ---- Recovered (R) ----
    ax = axes[2]
    ax.plot(mean_ca['timestep'], mean_ca['R_frac'], label='CA R', color='g', linestyle='--', linewidth=2)
    ax.fill_between(mean_ca['timestep'], mean_ca['R_frac'] - std_ca['R_frac'], mean_ca['R_frac'] + std_ca['R_frac'], color='g', alpha=0.2)
    ax.plot(ode_time, mean_ode_states[:, 2], label='ODE R', color='g', linewidth=2)
    ax.fill_between(ode_time, mean_ode_states[:, 2] - std_ode_states[:, 2], mean_ode_states[:, 2] + std_ode_states[:, 2], color='g', alpha=0.1)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Recovered Fraction', fontsize=12)
    ax.set_title('Recovered (R)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.suptitle(f"Waning={waning_val:.4f}, Mixing={mixing_val:.3f}, Infection={params['infection_prob']:.3f}", fontsize=14, y=0.995)
    
    # Save image
    img_filename = f"waning_mixing_infected{initial_infected_count}_waning{waning_val:.4f}_mixing{mixing_val:.3f}.png"
    img_path = os.path.join(img_dir, img_filename)
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"  Saved image: {img_path}")
    
    plt.show()

# ==========================================================
# Helper: save results to CSV
# ==========================================================
def save_results_to_csv(waning_val, mixing_val, mean_ca, std_ca, ode_time, mean_ode_states, std_ode_states, initial_infected_count, csv_dir):
    """Save CA and ODE results to CSV files."""
    csv_filename = f"waning_mixing_infected{initial_infected_count}_waning{waning_val:.4f}_mixing{mixing_val:.3f}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Time', 'CA_S', 'CA_S_std', 'CA_I', 'CA_I_std', 'CA_R', 'CA_R_std', 
                        'ODE_S', 'ODE_S_std', 'ODE_I', 'ODE_I_std', 'ODE_R', 'ODE_R_std'])
        
        # Determine max length (CA and ODE may have different time steps)
        ca_len = len(mean_ca['timestep'])
        ode_len = len(ode_time)
        max_len = max(ca_len, ode_len)
        
        # Write data rows
        for i in range(max_len):
            ca_time = mean_ca['timestep'][i] if i < ca_len else ''
            ca_s = mean_ca['S_frac'][i] if i < ca_len else ''
            ca_s_std = std_ca['S_frac'][i] if i < ca_len else ''
            ca_i = mean_ca['I_frac'][i] if i < ca_len else ''
            ca_i_std = std_ca['I_frac'][i] if i < ca_len else ''
            ca_r = mean_ca['R_frac'][i] if i < ca_len else ''
            ca_r_std = std_ca['R_frac'][i] if i < ca_len else ''
            
            ode_t = ode_time[i] if i < ode_len else ''
            ode_s = mean_ode_states[i, 0] if i < ode_len else ''
            ode_s_std = std_ode_states[i, 0] if i < ode_len else ''
            ode_i = mean_ode_states[i, 1] if i < ode_len else ''
            ode_i_std = std_ode_states[i, 1] if i < ode_len else ''
            ode_r = mean_ode_states[i, 2] if i < ode_len else ''
            ode_r_std = std_ode_states[i, 2] if i < ode_len else ''
            
            writer.writerow([ca_time, ca_s, ca_s_std, ca_i, ca_i_std, ca_r, ca_r_std,
                           ode_t, ode_s, ode_s_std, ode_i, ode_i_std, ode_r, ode_r_std])
    
    print(f"  Saved CSV: {csv_path}")

# ==========================================================
# Main experiment
# ==========================================================
def waning_mixing_infection_sensitivity_experiment():
    """Experiment: vary waning and mixing rates, use fixed infection probability from params."""
    total_cells = params['width'] * params['height']
    base_frac = compute_initial_infected_fraction(
        params['width'], params['height'], params['initial_infected_count']
    )
    print(f"Grid: {params['width']}x{params['height']} ({total_cells} cells)")
    print(f"Initial infected count: {params['initial_infected_count']}")
    print(f"Initial infected fraction (for ODE): {base_frac:.5f}\n")

    # Create output directories
    base_dir = os.path.dirname(__file__)
    csv_dir = os.path.join(base_dir, f"waning_mixing_csv_infected{params['initial_infected_count']}")
    img_dir = os.path.join(base_dir, f"waning_mixing_images_infected{params['initial_infected_count']}")
    
    Path(csv_dir).mkdir(parents=True, exist_ok=True)
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"CSV output directory: {csv_dir}")
    print(f"Image output directory: {img_dir}\n")

    # Loop through paired waning and mixing rates (changing together)
    for waning_val, mixing_val in waning_mixing_pairs:
        print(f"\n{'='*60}")
        print(f"Waning prob: {waning_val:.4f}, Mixing rate: {mixing_val:.3f}")
        print(f"Fixed infection_prob: {params['infection_prob']:.3f}")
        print(f"{'='*60}")

        ca_histories = []
        ode_states_list = []

        for sim_idx in range(params['num_simulations']):
            p = params.copy()
            p['waning_prob'] = waning_val
            p['mixing_rate'] = mixing_val

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

        print(f"  Completed for waning={waning_val:.4f}, mixing={mixing_val:.3f}")

        # Save CSV data
        save_results_to_csv(waning_val, mixing_val, mean_ca, std_ca, ode_time, mean_ode_states, std_ode_states, params['initial_infected_count'], csv_dir)

        # Plot comparison with error bars and save image
        plot_ca_ode_comparison(waning_val, mixing_val, mean_ca, std_ca, ode_time, mean_ode_states, std_ode_states, params['t_max'], params['initial_infected_count'], csv_dir, img_dir)



# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    waning_mixing_infection_sensitivity_experiment()