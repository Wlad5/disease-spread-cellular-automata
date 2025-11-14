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
    'width'                 : 20,
    'height'                : 20,
    'cell_size'             : 4,
    'initial_infected_count': 10,
    'mixing_rate'           : 0.00,
    'num_simulations'       : 2  # Number of simulations per parameter value
}

# Parameter range for mixing rate sensitivity analysis
mixing_rate_values = np.linspace(0.0, 1, 21)  # More frequent values
plots_per_figure = 5  # Show 5 plots per figure

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
# Updated Plotting with Error Bars (in batches)
# ==========================================================
def plot_comparison_with_error(param_name, param_values, ca_results, ode_results, t_max, ode_dt, plots_per_figure=5):
    num_params = len(param_values)
    num_figures = int(np.ceil(num_params / plots_per_figure))
    
    for fig_idx in range(num_figures):
        plt.figure(figsize=(16, 12))
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
            ax_s.set_title(f"S: {param_name}={val:.3f}", fontsize=11)
            ax_s.set_ylabel('Fraction')
            ax_s.set_ylim(0, 1)
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
            ax_i.set_title(f"I: {param_name}={val:.3f}", fontsize=11)
            ax_i.set_ylabel('Fraction')
            ax_i.set_ylim(0, 1)
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
            ax_r.set_title(f"R: {param_name}={val:.3f}", fontsize=11)
            ax_r.set_xlabel('Time')
            ax_r.set_ylabel('Fraction')
            ax_r.set_ylim(0, 1)
            if batch_pos == 0:
                ax_r.legend(loc='best', fontsize=9)
            if batch_pos > 0:
                ax_r.set_yticklabels([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"Parameter Sensitivity: {param_name} (Batch {fig_idx + 1}/{num_figures})", fontsize=16, y=0.98)
        plt.subplots_adjust(hspace=0.35, wspace=0.15)
        plt.show()

# ==========================================================
# Norm Calculation and Plotting
# ==========================================================
def calculate_norm(ca_results, ode_results):
    norms = {'S': [], 'I': [], 'R': []}
    for i in range(len(ca_results)):
        mean_ca, _ = ca_results[i]
        _, mean_ode_states, _ = ode_results[i]

        # Interpolate ODE results to match CA time steps
        ca_timesteps = mean_ca['timestep']
        ode_time = np.linspace(0, len(mean_ode_states) * params['ode_dt'], len(mean_ode_states))
        interp_ode_S = interp1d(ode_time, mean_ode_states[:, 0], kind='linear', fill_value="extrapolate")
        interp_ode_I = interp1d(ode_time, mean_ode_states[:, 1], kind='linear', fill_value="extrapolate")
        interp_ode_R = interp1d(ode_time, mean_ode_states[:, 2], kind='linear', fill_value="extrapolate")

        # Calculate L2 norms for S, I, and R
        norm_S = np.sqrt(np.sum((mean_ca['S_frac'] - interp_ode_S(ca_timesteps)) ** 2))
        norm_I = np.sqrt(np.sum((mean_ca['I_frac'] - interp_ode_I(ca_timesteps)) ** 2))
        norm_R = np.sqrt(np.sum((mean_ca['R_frac'] - interp_ode_R(ca_timesteps)) ** 2))

        norms['S'].append(norm_S)
        norms['I'].append(norm_I)
        norms['R'].append(norm_R)

    return norms

def plot_norm_vs_parameter(param_name, param_values, norms):
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, norms['S'], marker='o', linestyle='-', label='Norm (S)', color='b')
    plt.plot(param_values, norms['I'], marker='o', linestyle='-', label='Norm (I)', color='r')
    plt.plot(param_values, norms['R'], marker='o', linestyle='-', label='Norm (R)', color='g')
    plt.xlabel(param_name)
    plt.ylabel('Norm')
    plt.title(f'Norms for S, I, R vs {param_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================================
# Main experiment
# ==========================================================
def mixing_rate_sensitivity_experiment():
    total_cells = params['width'] * params['height']
    base_frac = compute_initial_infected_fraction(
        params['width'], params['height'], params['initial_infected_count']
    )
    print(f"Grid: {params['width']}x{params['height']} ({total_cells} cells)")
    print(f"Initial infected count: {params['initial_infected_count']}")
    print(f"Initial infected fraction (for ODE): {base_frac:.5f}\n")

    print(f"Running sensitivity for mixing_rate...")
    ca_results = []
    ode_results = []

    for val in mixing_rate_values:
        p = params.copy()
        p['mixing_rate'] = val

        ca_histories = []
        ode_states_list = []

        for _ in range(params['num_simulations']):
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
        std_ode_states  = np.std(ode_states_list, axis=0)

        print(f"  mixing_rate={val:.3f} → CA I₀={ca_initial_infected_frac:.5f}")

        ca_results.append((mean_ca, std_ca))
        ode_results.append((ode_time, mean_ode_states, std_ode_states))

    # Update the plotting function to handle mean ± std
    plot_comparison_with_error("mixing_rate", mixing_rate_values, ca_results, ode_results, params['t_max'], params['ode_dt'], plots_per_figure=5)

    # Calculate norms and plot norm vs parameter
    norms = calculate_norm(ca_results, ode_results)
    plot_norm_vs_parameter("mixing_rate", mixing_rate_values, norms)

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    mixing_rate_sensitivity_experiment()