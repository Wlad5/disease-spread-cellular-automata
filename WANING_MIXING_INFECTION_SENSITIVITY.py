import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util

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
    'initial_infected_count': 10,
    'mixing_rate'           : 0.00,
    'num_simulations'       : 1  # Number of simulations per parameter value
}

# Parameter ranges for waning and mixing rates (change together)
waning_prob_values = np.linspace(0.0, 1.0, 5)
mixing_rate_values = np.linspace(0.0, 1.0, 5)
waning_mixing_pairs = list(zip(waning_prob_values, mixing_rate_values))

# Parameter range for infection probability sensitivity analysis
infection_prob_values = np.linspace(0.0, 1.0, 5)

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
# Norm calculation and plotting
# ==========================================================
def compute_batch_norms(ca_results, ode_results, infection_values):
    """
    Compute L2 norm between CA and ODE trajectories for each infection probability.
    Returns lists of norms for S, I, R components and mean norm.
    """
    norms_S = []
    norms_I = []
    norms_R = []
    norms_mean = []
    
    for i, inf_val in enumerate(infection_values):
        mean_ca, std_ca = ca_results[i]
        ode_time, mean_ode_states, std_ode_states = ode_results[i]
        
        # Interpolate CA results to match ODE time points for fair comparison
        ca_times = np.array(mean_ca['timestep'])
        ca_S = np.array(mean_ca['S_frac'])
        ca_I = np.array(mean_ca['I_frac'])
        ca_R = np.array(mean_ca['R_frac'])
        
        # Interpolate CA to ODE time grid
        ca_S_interp = np.interp(ode_time, ca_times, ca_S)
        ca_I_interp = np.interp(ode_time, ca_times, ca_I)
        ca_R_interp = np.interp(ode_time, ca_times, ca_R)
        
        # Compute L2 norms
        norm_S = np.linalg.norm(ca_S_interp - mean_ode_states[:, 0])
        norm_I = np.linalg.norm(ca_I_interp - mean_ode_states[:, 1])
        norm_R = np.linalg.norm(ca_R_interp - mean_ode_states[:, 2])
        # Total norm: combine all differences into one vector and compute norm
        all_diffs = np.concatenate([
            ca_S_interp - mean_ode_states[:, 0],
            ca_I_interp - mean_ode_states[:, 1],
            ca_R_interp - mean_ode_states[:, 2]
        ])
        norm_total = np.linalg.norm(all_diffs)
        
        norms_S.append(norm_S)
        norms_I.append(norm_I)
        norms_R.append(norm_R)
        norms_mean.append(norm_total)
    
    return norms_S, norms_I, norms_R, norms_mean

def plot_batch_norms(infection_values, norms_S, norms_I, norms_R, norms_mean, waning_val, mixing_val):
    """Plot norms across infection probability parameter."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot individual component norms
    ax.plot(infection_values, norms_S, 'o-', label='L2 norm (S)', linewidth=2, markersize=8, color='blue')
    ax.plot(infection_values, norms_I, 's-', label='L2 norm (I)', linewidth=2, markersize=8, color='red')
    ax.plot(infection_values, norms_R, '^-', label='L2 norm (R)', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Infection Probability', fontsize=12)
    ax.set_ylabel('L2 Norm', fontsize=12)
    ax.set_title(f'Component Norms (Waning={waning_val:.4f}, Mixing={mixing_val:.3f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==========================================================
# Plotting with Error Bars
# ==========================================================
def plot_comparison_with_error(waning_val, mixing_val, infection_values, ca_results, ode_results, t_max, ode_dt):
    """Plot CA vs ODE for different infection probabilities using 3-row layout (S, I, R)."""
    plt.figure(figsize=(16, 12))
    num_params = len(infection_values)
    
    for i, inf_val in enumerate(infection_values):
        mean_ca, std_ca = ca_results[i]
        ode_time, mean_ode_states, std_ode_states = ode_results[i]

        # ---- Row 1: S (Susceptible) ----
        ax_s = plt.subplot(3, num_params, i + 1)
        ax_s.plot(mean_ca['timestep'], mean_ca['S_frac'], label='CA S', color='b', linestyle='--', linewidth=2)
        ax_s.fill_between(mean_ca['timestep'], mean_ca['S_frac'] - std_ca['S_frac'], mean_ca['S_frac'] + std_ca['S_frac'], color='b', alpha=0.2)
        ax_s.plot(ode_time, mean_ode_states[:, 0], label='ODE S', color='b', linewidth=2)
        ax_s.fill_between(ode_time, mean_ode_states[:, 0] - std_ode_states[:, 0], mean_ode_states[:, 0] + std_ode_states[:, 0], color='b', alpha=0.1)
        ax_s.set_title(f"S: infection_prob={inf_val:.3f}", fontsize=11)
        ax_s.set_ylabel('Fraction')
        ax_s.set_ylim(0, 1)
        if i == 0:
            ax_s.legend(loc='best', fontsize=9)
        if i > 0:
            ax_s.set_yticklabels([])

        # ---- Row 2: I (Infected) ----
        ax_i = plt.subplot(3, num_params, num_params + i + 1)
        ax_i.plot(mean_ca['timestep'], mean_ca['I_frac'], label='CA I', color='r', linestyle='--', linewidth=2)
        ax_i.fill_between(mean_ca['timestep'], mean_ca['I_frac'] - std_ca['I_frac'], mean_ca['I_frac'] + std_ca['I_frac'], color='r', alpha=0.2)
        ax_i.plot(ode_time, mean_ode_states[:, 1], label='ODE I', color='r', linewidth=2)
        ax_i.fill_between(ode_time, mean_ode_states[:, 1] - std_ode_states[:, 1], mean_ode_states[:, 1] + std_ode_states[:, 1], color='r', alpha=0.1)
        ax_i.set_title(f"I: infection_prob={inf_val:.3f}", fontsize=11)
        ax_i.set_ylabel('Fraction')
        ax_i.set_ylim(0, 1)
        if i == 0:
            ax_i.legend(loc='best', fontsize=9)
        if i > 0:
            ax_i.set_yticklabels([])

        # ---- Row 3: R (Recovered) ----
        ax_r = plt.subplot(3, num_params, 2 * num_params + i + 1)
        ax_r.plot(mean_ca['timestep'], mean_ca['R_frac'], label='CA R', color='g', linestyle='--', linewidth=2)
        ax_r.fill_between(mean_ca['timestep'], mean_ca['R_frac'] - std_ca['R_frac'], mean_ca['R_frac'] + std_ca['R_frac'], color='g', alpha=0.2)
        ax_r.plot(ode_time, mean_ode_states[:, 2], label='ODE R', color='g', linewidth=2)
        ax_r.fill_between(ode_time, mean_ode_states[:, 2] - std_ode_states[:, 2], mean_ode_states[:, 2] + std_ode_states[:, 2], color='g', alpha=0.1)
        ax_r.set_title(f"R: infection_prob={inf_val:.3f}", fontsize=11)
        ax_r.set_xlabel('Time')
        ax_r.set_ylabel('Fraction')
        ax_r.set_ylim(0, 1)
        if i == 0:
            ax_r.legend(loc='best', fontsize=9)
        if i > 0:
            ax_r.set_yticklabels([])

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.suptitle(f"Parameter Sensitivity: infection_prob (Waning={waning_val:.4f}, Mixing={mixing_val:.3f})", fontsize=16, y=0.995)
    plt.show()

# ==========================================================
# Main experiment
# ==========================================================
def waning_mixing_infection_sensitivity_experiment():
    """Experiment: vary waning and mixing, then run infection sensitivity for each combination."""
    total_cells = params['width'] * params['height']
    base_frac = compute_initial_infected_fraction(
        params['width'], params['height'], params['initial_infected_count']
    )
    print(f"Grid: {params['width']}x{params['height']} ({total_cells} cells)")
    print(f"Initial infected count: {params['initial_infected_count']}")
    print(f"Initial infected fraction (for ODE): {base_frac:.5f}\n")

    # Loop through paired waning and mixing rates (changing together)
    for waning_val, mixing_val in waning_mixing_pairs:
        print(f"\n{'='*60}")
        print(f"Waning prob: {waning_val:.4f}, Mixing rate: {mixing_val:.3f}")
        print(f"Varying infection_prob: {infection_prob_values}")
        print(f"{'='*60}")

        ca_results = []
        ode_results = []

        for inf_val in infection_prob_values:
            p = params.copy()
            p['infection_prob'] = inf_val
            p['waning_prob'] = waning_val
            p['mixing_rate'] = mixing_val

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

            print(f"  infection_prob={inf_val:.3f} → Completed")

            ca_results.append((mean_ca, std_ca))
            ode_results.append((ode_time, mean_ode_states, std_ode_states))

        # Plot results for this waning/mixing combination
        plot_comparison_with_error(waning_val, mixing_val, infection_prob_values, ca_results, ode_results, params['t_max'], params['ode_dt'])
        
        # Compute and plot batch norms
        norms_S, norms_I, norms_R, norms_mean = compute_batch_norms(ca_results, ode_results, infection_prob_values)
        plot_batch_norms(infection_prob_values, norms_S, norms_I, norms_R, norms_mean, waning_val, mixing_val)

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    waning_mixing_infection_sensitivity_experiment()