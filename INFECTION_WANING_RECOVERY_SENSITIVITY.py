"""
Sensitivity Analysis: Infection + Waning (varied together), then Recovery

In this analysis:
- infection_prob and waning_prob are varied together
- recovery_prob is varied for multiple values of the combined parameter
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
# Directory setup
# ==========================================================
def setup_directories(base_name):
    """Create output directories for CSV and images"""
    csv_dir = os.path.join(os.path.dirname(__file__), f"{base_name}_csv")
    img_dir = os.path.join(os.path.dirname(__file__), f"{base_name}_images")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    return csv_dir, img_dir

csv_dir, img_dir = setup_directories("infection_waning_recovery")

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
    'width'                 : 100,
    'height'                : 100,
    'cell_size'             : 4,
    'initial_infected_count': 10,
    'mixing_rate'           : 0.00,
    'num_simulations'       : 1
}

# Parameter ranges
# Infection + Waning varied together using linspace
infection_prob_range = np.linspace(0.0, 1, 20)
waning_prob_range = np.linspace(0.0, 1, 20)
infection_waning_pairs = list(zip(infection_prob_range, waning_prob_range))
# Recovery varied for each pair
recovery_prob_values = np.linspace(0.0, 1, 20)

plots_per_figure = 5

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

    sim.grid.infect_random(n=initial_infected_count)
    initial_infected_frac = compute_initial_infected_fraction(width, height, initial_infected_count)

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
# Plotting with Error Bars (in batches)
# ==========================================================
def plot_comparison_with_error(iw_pair_idx, iw_pair, recovery_values, ca_results, ode_results, 
                               t_max, ode_dt, plots_per_figure=3):
    """
    Plot comparison between CA and ODE for fixed infection+waning pair,
    varying recovery probability. Save plots to disk.
    """
    num_recovery = len(recovery_values)
    num_figures = int(np.ceil(num_recovery / plots_per_figure))
    
    inf_prob, wan_prob = iw_pair
    
    for fig_idx in range(num_figures):
        fig = plt.figure(figsize=(16, 12))
        start_idx = fig_idx * plots_per_figure
        end_idx = min(start_idx + plots_per_figure, num_recovery)
        batch_size = end_idx - start_idx
        
        for batch_pos, i in enumerate(range(start_idx, end_idx)):
            val = recovery_values[i]
            mean_ca, std_ca = ca_results[iw_pair_idx][i]
            ode_time, mean_ode_states, std_ode_states = ode_results[iw_pair_idx][i]

            # ---- Row 1: S (Susceptible) ----
            ax_s = plt.subplot(3, batch_size, batch_pos + 1)
            ax_s.plot(mean_ca['timestep'], mean_ca['S_frac'], label='CA S', color='b', linestyle='--', linewidth=2)
            ax_s.fill_between(mean_ca['timestep'], mean_ca['S_frac'] - std_ca['S_frac'], 
                             mean_ca['S_frac'] + std_ca['S_frac'], color='b', alpha=0.2)
            ax_s.plot(ode_time, mean_ode_states[:, 0], label='ODE S', color='b', linewidth=2)
            ax_s.fill_between(ode_time, mean_ode_states[:, 0] - std_ode_states[:, 0], 
                             mean_ode_states[:, 0] + std_ode_states[:, 0], color='b', alpha=0.1)
            ax_s.set_title(f"S: recovery={val:.4f}", fontsize=11)
            ax_s.set_ylabel('Fraction')
            ax_s.set_ylim(0, 1)
            if batch_pos == 0:
                ax_s.legend(loc='best', fontsize=9)
            if batch_pos > 0:
                ax_s.set_yticklabels([])

            # ---- Row 2: I (Infected) ----
            ax_i = plt.subplot(3, batch_size, batch_size + batch_pos + 1)
            ax_i.plot(mean_ca['timestep'], mean_ca['I_frac'], label='CA I', color='r', linestyle='--', linewidth=2)
            ax_i.fill_between(mean_ca['timestep'], mean_ca['I_frac'] - std_ca['I_frac'], 
                             mean_ca['I_frac'] + std_ca['I_frac'], color='r', alpha=0.2)
            ax_i.plot(ode_time, mean_ode_states[:, 1], label='ODE I', color='r', linewidth=2)
            ax_i.fill_between(ode_time, mean_ode_states[:, 1] - std_ode_states[:, 1], 
                             mean_ode_states[:, 1] + std_ode_states[:, 1], color='r', alpha=0.1)
            ax_i.set_title(f"I: recovery={val:.4f}", fontsize=11)
            ax_i.set_ylabel('Fraction')
            ax_i.set_ylim(0, 1)
            if batch_pos == 0:
                ax_i.legend(loc='best', fontsize=9)
            if batch_pos > 0:
                ax_i.set_yticklabels([])

            # ---- Row 3: R (Recovered) ----
            ax_r = plt.subplot(3, batch_size, 2 * batch_size + batch_pos + 1)
            ax_r.plot(mean_ca['timestep'], mean_ca['R_frac'], label='CA R', color='g', linestyle='--', linewidth=2)
            ax_r.fill_between(mean_ca['timestep'], mean_ca['R_frac'] - std_ca['R_frac'], 
                             mean_ca['R_frac'] + std_ca['R_frac'], color='g', alpha=0.2)
            ax_r.plot(ode_time, mean_ode_states[:, 2], label='ODE R', color='g', linewidth=2)
            ax_r.fill_between(ode_time, mean_ode_states[:, 2] - std_ode_states[:, 2], 
                             mean_ode_states[:, 2] + std_ode_states[:, 2], color='g', alpha=0.1)
            ax_r.set_title(f"R: recovery={val:.4f}", fontsize=11)
            ax_r.set_xlabel('Time')
            ax_r.set_ylabel('Fraction')
            ax_r.set_ylim(0, 1)
            if batch_pos == 0:
                ax_r.legend(loc='best', fontsize=9)
            if batch_pos > 0:
                ax_r.set_yticklabels([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"Infection+Waning (inf={inf_prob:.3f}, wan={wan_prob:.4f}) vs Recovery " +
                    f"(Batch {fig_idx + 1}/{num_figures})", fontsize=16, y=0.98)
        plt.subplots_adjust(hspace=0.35, wspace=0.15)
        
        # Save figure instead of showing
        filename = os.path.join(img_dir, f"iw_pair_{iw_pair_idx}_batch_{fig_idx}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved figure: {filename}")

# ==========================================================
# Norm Calculation
# ==========================================================
def calculate_norm(ca_results, ode_results):
    """Calculate L2 norms between CA and ODE results"""
    norms_by_pair = []
    
    for pair_idx in range(len(ca_results)):
        norms = {'S': [], 'I': [], 'R': []}
        for i in range(len(ca_results[pair_idx])):
            mean_ca, _ = ca_results[pair_idx][i]
            _, mean_ode_states, _ = ode_results[pair_idx][i]

            ca_timesteps = mean_ca['timestep']
            ode_time = np.linspace(0, len(mean_ode_states) * params['ode_dt'], len(mean_ode_states))
            interp_ode_S = interp1d(ode_time, mean_ode_states[:, 0], kind='linear', fill_value="extrapolate")
            interp_ode_I = interp1d(ode_time, mean_ode_states[:, 1], kind='linear', fill_value="extrapolate")
            interp_ode_R = interp1d(ode_time, mean_ode_states[:, 2], kind='linear', fill_value="extrapolate")

            norm_S = np.sqrt(np.sum((mean_ca['S_frac'] - interp_ode_S(ca_timesteps)) ** 2))
            norm_I = np.sqrt(np.sum((mean_ca['I_frac'] - interp_ode_I(ca_timesteps)) ** 2))
            norm_R = np.sqrt(np.sum((mean_ca['R_frac'] - interp_ode_R(ca_timesteps)) ** 2))

            norms['S'].append(norm_S)
            norms['I'].append(norm_I)
            norms['R'].append(norm_R)
        
        norms_by_pair.append(norms)
    
    return norms_by_pair

def plot_norm_vs_parameter(recovery_values, norms_by_pair, infection_waning_pairs):
    """Plot norms for each infection+waning pair and save separately"""
    for pair_idx, (inf_prob, wan_prob) in enumerate(infection_waning_pairs):
        fig = plt.figure(figsize=(10, 6))
        norms = norms_by_pair[pair_idx]
        plt.plot(recovery_values, norms['S'], marker='o', linestyle='-', label='Norm (S)', color='b', linewidth=2)
        plt.plot(recovery_values, norms['I'], marker='o', linestyle='-', label='Norm (I)', color='r', linewidth=2)
        plt.plot(recovery_values, norms['R'], marker='o', linestyle='-', label='Norm (R)', color='g', linewidth=2)
        plt.xlabel('Recovery Probability', fontsize=12)
        plt.ylabel('L2 Norm', fontsize=12)
        plt.title(f'Norms vs Recovery (inf={inf_prob:.3f}, wan={wan_prob:.4f})', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(img_dir, f"norm_iw_pair_{pair_idx}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved norm plot: {filename}")

def save_results_to_csv(iw_pair_idx, iw_pair, recovery_values, ca_results, ode_results):
    """Save CA and ODE results to CSV files"""
    inf_prob, wan_prob = iw_pair
    
    for i, recovery_val in enumerate(recovery_values):
        mean_ca, std_ca = ca_results[iw_pair_idx][i]
        ode_time, mean_ode_states, std_ode_states = ode_results[iw_pair_idx][i]
        
        # CA results
        ca_filename = os.path.join(csv_dir, f"ca_iw_{iw_pair_idx}_recovery_{i}.csv")
        with open(ca_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'S_frac', 'S_std', 'I_frac', 'I_std', 'R_frac', 'R_std'])
            for j, t in enumerate(mean_ca['timestep']):
                writer.writerow([t, mean_ca['S_frac'][j], std_ca['S_frac'][j],
                               mean_ca['I_frac'][j], std_ca['I_frac'][j],
                               mean_ca['R_frac'][j], std_ca['R_frac'][j]])
        
        # ODE results
        ode_filename = os.path.join(csv_dir, f"ode_iw_{iw_pair_idx}_recovery_{i}.csv")
        with open(ode_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'S', 'S_std', 'I', 'I_std', 'R', 'R_std'])
            for j, t in enumerate(ode_time):
                writer.writerow([t, mean_ode_states[j, 0], std_ode_states[j, 0],
                               mean_ode_states[j, 1], std_ode_states[j, 1],
                               mean_ode_states[j, 2], std_ode_states[j, 2]])

def save_norms_to_csv(recovery_values, norms_by_pair, infection_waning_pairs):
    """Save norm values to CSV"""
    norms_filename = os.path.join(csv_dir, "norms_summary.csv")
    with open(norms_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iw_pair_idx', 'inf_prob', 'wan_prob', 'recovery_val', 'norm_S', 'norm_I', 'norm_R'])
        
        for pair_idx, (inf_prob, wan_prob) in enumerate(infection_waning_pairs):
            norms = norms_by_pair[pair_idx]
            for i, recovery_val in enumerate(recovery_values):
                writer.writerow([pair_idx, inf_prob, wan_prob, recovery_val,
                               norms['S'][i], norms['I'][i], norms['R'][i]])

# ==========================================================
# Main experiment
# ==========================================================
def sensitivity_experiment():
    total_cells = params['width'] * params['height']
    base_frac = compute_initial_infected_fraction(
        params['width'], params['height'], params['initial_infected_count']
    )
    print(f"Grid: {params['width']}x{params['height']} ({total_cells} cells)")
    print(f"Initial infected count: {params['initial_infected_count']}")
    print(f"Initial infected fraction (for ODE): {base_frac:.5f}\n")

    print(f"Running sensitivity for Infection+Waning, then Recovery...")
    all_ca_results = []
    all_ode_results = []

    for iw_idx, (inf_prob, wan_prob) in enumerate(infection_waning_pairs):
        print(f"\n--- Infection={inf_prob:.3f}, Waning={wan_prob:.4f} ---")
        ca_results = []
        ode_results = []

        for recovery_val in recovery_prob_values:
            p = params.copy()
            p['infection_prob'] = inf_prob
            p['recovery_prob'] = recovery_val
            p['waning_prob'] = wan_prob

            ca_histories = []
            ode_states_list = []

            for _ in range(params['num_simulations']):
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

            # Calculate mean and std
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

            mean_ode_states = np.mean(ode_states_list, axis=0)
            std_ode_states = np.std(ode_states_list, axis=0)

            print(f"  recovery={recovery_val:.4f} → CA I₀={ca_initial_infected_frac:.5f}")

            ca_results.append((mean_ca, std_ca))
            ode_results.append((ode_time, mean_ode_states, std_ode_states))

        all_ca_results.append(ca_results)
        all_ode_results.append(ode_results)
        
        # Save results to CSV
        save_results_to_csv(iw_idx, (inf_prob, wan_prob), recovery_prob_values, all_ca_results, all_ode_results)
        
        # Plot for this pair
        plot_comparison_with_error(iw_idx, (inf_prob, wan_prob), recovery_prob_values, 
                                  all_ca_results, all_ode_results, params['t_max'], params['ode_dt'], plots_per_figure)

    # Calculate and plot norms
    norms_by_pair = calculate_norm(all_ca_results, all_ode_results)
    save_norms_to_csv(recovery_prob_values, norms_by_pair, infection_waning_pairs)
    plot_norm_vs_parameter(recovery_prob_values, norms_by_pair, infection_waning_pairs)
    
    print(f"\nAll results saved to:")
    print(f"  CSV files: {csv_dir}")
    print(f"  Images: {img_dir}")

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    sensitivity_experiment()
