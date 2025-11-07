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
    'mixing_rate'           : 0.00
}

# Parameter ranges for sensitivity analysis
param_ranges = {
    'infection_prob': np.linspace(0.0, 1, 5),
    'recovery_prob' : np.linspace(0.0, 1, 5),
    'waning_prob'   : np.linspace(0.0, 1, 5),
    'mixing_rate'   : np.linspace(0.0, 1, 5),
}

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
# Plotting
# ==========================================================
def plot_comparison(param_name, param_values, ca_results, ode_results, t_max, ode_dt):
    plt.figure(figsize=(16, 10))
    for i, val in enumerate(param_values):
        ca_hist = ca_results[i]
        ode_time, ode_states = ode_results[i]

        # Interpolate ODE to CA timesteps for norm calculation
        ca_timesteps = np.array(ca_hist['timestep'])
        ode_S_interp = np.interp(ca_timesteps, ode_time, ode_states[:, 0])
        ode_I_interp = np.interp(ca_timesteps, ode_time, ode_states[:, 1])
        ode_R_interp = np.interp(ca_timesteps, ode_time, ode_states[:, 2])

        # Calculate norms (L2 norm)
        norm_S = np.linalg.norm(np.array(ca_hist['S_frac']) - ode_S_interp)
        norm_I = np.linalg.norm(np.array(ca_hist['I_frac']) - ode_I_interp)
        norm_R = np.linalg.norm(np.array(ca_hist['R_frac']) - ode_R_interp)

        # Calculate peak infection (max infected fraction)
        peak_I_ca = np.max(ca_hist['I_frac'])
        peak_I_ode = np.max(ode_states[:, 1])

        # ---- CA plots ----
        ax_ca = plt.subplot(2, len(param_values), i + 1)
        ax_ca.plot(ca_hist['timestep'], ca_hist['S_frac'], label='CA S', color='b', linestyle='--')
        ax_ca.plot(ca_hist['timestep'], ca_hist['I_frac'], label='CA I', color='r', linestyle='--')
        ax_ca.plot(ca_hist['timestep'], ca_hist['R_frac'], label='CA R', color='g', linestyle='--')
        ax_ca.set_title(f"CA: {param_name}={val:.3f}")
        ax_ca.set_xlabel('Time')
        ax_ca.set_ylabel('Fraction')
        ax_ca.set_ylim(0, 1)
        if i == 0:
            ax_ca.legend()
        # Show metrics below the plot
        metrics_text = (f"Norm S: {norm_S:.4f}\nNorm I: {norm_I:.4f}\nNorm R: {norm_R:.4f}\n"
                       f"Peak I (CA): {peak_I_ca:.4f}")
        ax_ca.text(0.02, -0.35, metrics_text, transform=ax_ca.transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ---- ODE plots ----
        ax_ode = plt.subplot(2, len(param_values), len(param_values) + i + 1)
        ax_ode.plot(ode_time, ode_states[:, 0], label='ODE S', color='b')
        ax_ode.plot(ode_time, ode_states[:, 1], label='ODE I', color='r')
        ax_ode.plot(ode_time, ode_states[:, 2], label='ODE R', color='g')
        ax_ode.set_title(f"ODE: {param_name}={val:.3f}")
        ax_ode.set_xlabel('Time')
        ax_ode.set_ylabel('Fraction')
        ax_ode.set_ylim(0, 1)
        if i == 0:
            ax_ode.legend()
        # Show metrics below the plot
        metrics_text_ode = f"Peak I (ODE): {peak_I_ode:.4f}"
        ax_ode.text(0.02, -0.18, metrics_text_ode, transform=ax_ode.transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Increased bottom margin
    plt.suptitle(f"Parameter Sensitivity: {param_name}", fontsize=16, y=1.02)
    
    # Add parameter information at the bottom
    param_text = (f"Grid={params['width']}x{params['height']}, "
                 f"Initial infected={params['initial_infected_count']}, "
                 f"k={params['k']}, dt={params['delta_t']}, "
                 f"infection={params['infection_prob']:.3f}, "
                 f"recovery={params['recovery_prob']:.3f}, "
                 f"waning={params['waning_prob']:.3f}, "
                 f"mixing={params['mixing_rate']:.3f}")
    plt.figtext(0.5, 0.02, param_text, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.show()

# ==========================================================
# Main experiment
# ==========================================================
def parameter_sensitivity_experiment():
    total_cells = params['width'] * params['height']
    base_frac = compute_initial_infected_fraction(
        params['width'], params['height'], params['initial_infected_count']
    )
    print(f"Grid: {params['width']}x{params['height']} ({total_cells} cells)")
    print(f"Initial infected count: {params['initial_infected_count']}")
    print(f"Initial infected fraction (for ODE): {base_frac:.5f}\n")

    for param_name, values in param_ranges.items():
        print(f"Running sensitivity for {param_name}...")
        ca_results = []
        ode_results = []

        for val in values:
            p = params.copy()
            p[param_name] = val

            if param_name == 'mixing_rate':
                # Vary mixing_rate in CA, keep ODE fixed
                ca_hist, ca_initial_infected_frac = run_ca_simulation(
                    infection_prob=p['infection_prob'],
                    recovery_prob=p['recovery_prob'],
                    waning_prob=p['waning_prob'],
                    t_max=p['t_max'],
                    width=p['width'],
                    height=p['height'],
                    cell_size=p['cell_size'],
                    initial_infected_count=p['initial_infected_count'],
                    mixing_rate=val  # Vary mixing_rate here
                )
                # ODE: use default mixing_rate (not used), just run once for reference
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

            print(f"  {param_name}={val:.3f} → CA I₀={ca_initial_infected_frac:.5f}")

            ca_results.append(ca_hist)
            ode_results.append((ode_time, ode_states))

        plot_comparison(param_name, values, ca_results, ode_results, params['t_max'], params['ode_dt'])

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    parameter_sensitivity_experiment()