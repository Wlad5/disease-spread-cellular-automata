import matplotlib.pyplot as plt
import numpy as np
from SIRS_ODE_SOLVER import solve_sirs_from_ca_params
from SIRS import Simulation

def run_ca_simulation(width, height, infection_prob, recovery_prob, waning_prob, delta_t, steps=500, initial_infected=8):
    # Headless CA simulation: no pygame window
    from SIRS import Grid
    grid = Grid(width, height, infection_prob, recovery_prob, waning_prob, delta_t)
    grid.infect_random(n=initial_infected)
    for _ in range(steps):
        grid.update()
    history = grid.history
    return history

def plot_comparison(ca_history, ode_time, ode_states, title, params):
    plt.figure(figsize=(12, 7))  # Made figure slightly taller to accommodate parameters
    # CA
    plt.plot(ca_history['timestep'], ca_history['S_frac'], 'b--', label='CA Susceptible')
    plt.plot(ca_history['timestep'], ca_history['I_frac'], 'r--', label='CA Infected')
    plt.plot(ca_history['timestep'], ca_history['R_frac'], 'g--', label='CA Recovered')
    # ODE
    plt.plot(ode_time, ode_states[:, 0], 'b-', label='ODE Susceptible')
    plt.plot(ode_time, ode_states[:, 1], 'r-', label='ODE Infected')
    plt.plot(ode_time, ode_states[:, 2], 'g-', label='ODE Recovered')
    plt.xlabel('Time step')
    plt.ylabel('Fraction of Population')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add parameters text at the bottom
    param_text = f"Parameters: infection_prob={params['infection_prob']}, recovery_prob={params['recovery_prob']}, "
    param_text += f"waning_prob={params['waning_prob']}, k={params['k']}, delta_t={params['delta_t']}, "
    param_text += f"initial_infected={params['initial_infected']}"
    plt.figtext(0.1, 0.02, param_text, wrap=True, horizontalalignment='left', fontsize=8)
    
    plt.tight_layout()
    # Adjust layout to make room for parameter text
    plt.subplots_adjust(bottom=0.2)
    plt.show()

def main():
    infection_prob = 0.08
    recovery_prob = 0.1
    waning_prob = 0.002
    delta_t = 1.0
    k = 8
    steps = 500
    initial_infected_cells = 10
    ca_grid_sizes = [(50, 50), (100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
    for width, height in ca_grid_sizes:
        ode_initial_infected = initial_infected_cells / (width * height)
        ode_time, ode_states, ode_params = solve_sirs_from_ca_params(
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
            waning_prob=waning_prob,
            k=k,
            delta_t=delta_t,
            initial_infected=ode_initial_infected,
            t_max=steps,
            dt=1.0
        )
        ca_history = run_ca_simulation(width, height, infection_prob, recovery_prob, waning_prob, delta_t, steps, initial_infected=initial_infected_cells)
        
        params = {
            'infection_prob': infection_prob,
            'recovery_prob': recovery_prob,
            'waning_prob': waning_prob,
            'k': k,
            'delta_t': delta_t,
            'initial_infected': initial_infected_cells
        }
        plot_comparison(ca_history, ode_time, ode_states, f"SIRS CA vs ODE (Grid: {width}x{height})", params)

if __name__ == "__main__":
    main()
