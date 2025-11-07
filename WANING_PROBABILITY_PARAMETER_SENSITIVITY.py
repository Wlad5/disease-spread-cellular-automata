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
    'num_simulations'       : 5  # Number of simulations per parameter value
}

# Parameter range for waning probability sensitivity analysis
waning_prob_values = np.linspace(0.0, 1, 5)

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
# Updated Plotting with Error Bars
# ==========================================================
def plot_comparison_with_error(param_name, param_values, ca_results, ode_results, t_max, ode_dt):
    plt.figure(figsize=(16, 10))
    for i, val in enumerate(param_values):
        mean_ca, std_ca = ca_results[i]
        ode_time, mean_ode_states, std_ode_states = ode_results[i]

        # ---- CA plots ----
        ax_ca = plt.subplot(2, len(param_values), i + 1)
        ax_ca.plot(mean_ca['timestep'], mean_ca['S_frac'], label='CA S', color='b', linestyle='--')
        ax_ca.fill_between(mean_ca['timestep'], mean_ca['S_frac'] - std_ca['S_frac'], mean_ca['S_frac'] + std_ca['S_frac'], color='b', alpha=0.2)
        ax_ca.plot(mean_ca['timestep'], mean_ca['I_frac'], label='CA I', color='r', linestyle='--')
        ax_ca.fill_between(mean_ca['timestep'], mean_ca['I_frac'] - std_ca['I_frac'], mean_ca['I_frac'] + std_ca['I_frac'], color='r', alpha=0.2)
        ax_ca.plot(mean_ca['timestep'], mean_ca['R_frac'], label='CA R', color='g', linestyle='--')
        ax_ca.fill_between(mean_ca['timestep'], mean_ca['R_frac'] - std_ca['R_frac'], mean_ca['R_frac'] + std_ca['R_frac'], color='g', alpha=0.2)
        ax_ca.set_title(f"CA: {param_name}={val:.3f}")
        ax_ca.set_xlabel('Time')
        ax_ca.set_ylabel('Fraction')
        ax_ca.set_ylim(0, 1)
        if i == 0:
            ax_ca.legend()

        # ---- ODE plots ----
        ax_ode = plt.subplot(2, len(param_values), len(param_values) + i + 1)
        ax_ode.plot(ode_time, mean_ode_states[:, 0], label='ODE S', color='b')
        ax_ode.fill_between(ode_time, mean_ode_states[:, 0] - std_ode_states[:, 0], mean_ode_states[:, 0] + std_ode_states[:, 0], color='b', alpha=0.2)
        ax_ode.plot(ode_time, mean_ode_states[:, 1], label='ODE I', color='r')
        ax_ode.fill_between(ode_time, mean_ode_states[:, 1] - std_ode_states[:, 1], mean_ode_states[:, 1] + std_ode_states[:, 1], color='r', alpha=0.2)
        ax_ode.plot(ode_time, mean_ode_states[:, 2], label='ODE R', color='g')
        ax_ode.fill_between(ode_time, mean_ode_states[:, 2] - std_ode_states[:, 2], mean_ode_states[:, 2] + std_ode_states[:, 2], color='g', alpha=0.2)
        ax_ode.set_title(f"ODE: {param_name}={val:.3f}")
        ax_ode.set_xlabel('Time')
        ax_ode.set_ylabel('Fraction')
        ax_ode.set_ylim(0, 1)
        if i == 0:
            ax_ode.legend()

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.suptitle(f"Parameter Sensitivity: {param_name}", fontsize=16, y=1.02)
    plt.show()

# ==========================================================
# Main experiment
# ==========================================================
def waning_prob_sensitivity_experiment():
    total_cells = params['width'] * params['height']
    base_frac = compute_initial_infected_fraction(
        params['width'], params['height'], params['initial_infected_count']
    )
    print(f"Grid: {params['width']}x{params['height']} ({total_cells} cells)")
    print(f"Initial infected count: {params['initial_infected_count']}")
    print(f"Initial infected fraction (for ODE): {base_frac:.5f}\n")

    print(f"Running sensitivity for waning_prob...")
    ca_results = []
    ode_results = []

    for val in waning_prob_values:
        p = params.copy()
        p['waning_prob'] = val

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

        # Calculate mean and std for CA results
        mean_ca = {
            'S_frac': np.mean([np.array(hist['S_frac']) for hist in ca_histories], axis=0),
            'I_frac': np.mean([np.array(hist['I_frac']) for hist in ca_histories], axis=0),
            'R_frac': np.mean([np.array(hist['R_frac']) for hist in ca_histories], axis=0),
            'timestep': ca_histories[0]['timestep']  # Timesteps are the same for all runs
        }
        std_ca = {
            'S_frac': np.std([np.array(hist['S_frac']) for hist in ca_histories], axis=0),
            'I_frac': np.std([np.array(hist['I_frac']) for hist in ca_histories], axis=0),
            'R_frac': np.std([np.array(hist['R_frac']) for hist in ca_histories], axis=0),
        }

        # Calculate mean and std for ODE results
        mean_ode_states = np.mean(ode_states_list, axis=0)
        std_ode_states = np.std(ode_states_list, axis=0)

        print(f"  waning_prob={val:.3f} → CA I₀={ca_initial_infected_frac:.5f}")

        ca_results.append((mean_ca, std_ca))
        ode_results.append((ode_time, mean_ode_states, std_ode_states))

    # Update the plotting function to handle mean ± std
    plot_comparison_with_error("waning_prob", waning_prob_values, ca_results, ode_results, params['t_max'], params['ode_dt'])

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    waning_prob_sensitivity_experiment()