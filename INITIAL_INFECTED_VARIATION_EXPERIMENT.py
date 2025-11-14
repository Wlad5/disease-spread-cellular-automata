"""
Initial Infected Cells Variation Experiment
This script varies the number of initially infected cells and runs multiple simulations
for each initial condition, comparing CA results with ODE solutions.

Results are displayed in a 2-row plot grid:
  - Row 1: CA results (one subplot per initial infected count)
  - Row 2: ODE results (one subplot per initial infected count)
  
Each subplot shows the mean and std of multiple runs for that initial infected count.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Dict, Tuple
import pygame
from SIRS import Grid
from SIRS_ODE_SOLVER import solve_sirs_from_ca_params


class InitialInfectedVariationExperiment:
    def __init__(self,
                 width: int = 100,
                 height: int = 100,
                 infection_prob: float = 0.08,
                 recovery_prob: float = 0.1,
                 waning_prob: float = 0.002,
                 delta_t: float = 1.0,
                 mixing_rate: float = 0.00):
        self.width = width
        self.height = height
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.waning_prob = waning_prob
        self.delta_t = delta_t
        self.mixing_rate = mixing_rate

        # storage: keyed by initial_infected_count
        self.ca_results: Dict[int, Dict] = {}  # {initial_count: {'runs': [...], 'avg': df, 'std': df}}
        self.ode_results: Dict[int, pd.DataFrame] = {}  # {initial_count: df}

    def run_ca_once(self, initial_infected_count: int, max_steps: int = 5000) -> pd.DataFrame:
        """Run a single CA simulation and return timestep series."""
        pygame.init()

        grid = Grid(self.width, self.height,
                    self.infection_prob, self.recovery_prob,
                    self.waning_prob, self.delta_t,
                    self.mixing_rate)
        grid.infect_random(n=initial_infected_count)

        for step in range(max_steps):
            grid.update()
            _, I_frac, _ = grid.get_population_fractions()
            # # early stop if epidemic died out
            # if step > 100 and I_frac < 0.0001:
            #     break

        # record final fractions
        grid.record_fractions()

        # quit pygame for this run
        try:
            pygame.quit()
        except Exception:
            pass

        # build DataFrame
        history = grid.history
        df = pd.DataFrame({
            'timestep': history['timestep'],
            'S_frac': history['S_frac'],
            'I_frac': history['I_frac'],
            'R_frac': history['R_frac']
        })
        return df

    def run_ca_for_initial_count(self, initial_infected_count: int, n_runs: int = 10, max_steps: int = 5000):
        """Run n_runs CA simulations for a specific initial infected count."""
        print(f"\n  Running {n_runs} CA simulations with initial_infected={initial_infected_count}...")
        all_runs = []
        for i in range(n_runs):
            print(f"    ▶️ Run {i+1}/{n_runs}")
            df = self.run_ca_once(initial_infected_count=initial_infected_count, max_steps=max_steps)
            df = df.copy()
            df['run'] = i + 1
            all_runs.append(df)

        # Combine runs
        combined = pd.concat(all_runs, ignore_index=True)

        # Compute mean and std grouped by timestep
        avg = combined.groupby('timestep')[['S_frac', 'I_frac', 'R_frac']].mean().reset_index()
        std = combined.groupby('timestep')[['S_frac', 'I_frac', 'R_frac']].std().reset_index()

        # Store results
        self.ca_results[initial_infected_count] = {
            'runs': all_runs,
            'avg': avg,
            'std': std
        }

        # Save to CSV
        combined.to_csv(f'sirs_ca_initial_{initial_infected_count}.csv', index=False)
        avg.to_csv(f'sirs_ca_avg_initial_{initial_infected_count}.csv', index=False)
        print(f"  Saved sirs_ca_initial_{initial_infected_count}.csv and sirs_ca_avg_initial_{initial_infected_count}.csv")

    def run_ode_for_initial_count(self, initial_infected_count: int, dt: float = 0.1):
        """Run ODE solver for a specific initial infected count."""
        if initial_infected_count not in self.ca_results:
            print(f"Warning: No CA results for initial_infected={initial_infected_count}, skipping ODE.")
            return

        # Determine t_max from CA results
        ca_avg = self.ca_results[initial_infected_count]['avg']
        t_max = float(ca_avg['timestep'].max())

        # Calculate initial_infected as fraction
        initial_infected_frac = initial_infected_count / (self.width * self.height)

        print(f"  Running ODE solver with initial_infected={initial_infected_count} (fraction={initial_infected_frac:.4f})...")

        time_points, states, params = solve_sirs_from_ca_params(
            infection_prob=self.infection_prob,
            recovery_prob=self.recovery_prob,
            waning_prob=self.waning_prob,
            k=8,
            delta_t=self.delta_t,
            initial_infected=initial_infected_frac,
            t_max=t_max,
            dt=dt
        )

        ode_df = pd.DataFrame({
            'time': time_points,
            'S': states[:, 0],
            'I': states[:, 1],
            'R': states[:, 2]
        })

        self.ode_results[initial_infected_count] = ode_df

        # Save to CSV
        ode_df.to_csv(f'sirs_ode_initial_{initial_infected_count}.csv', index=False)
        print(f"  Saved sirs_ode_initial_{initial_infected_count}.csv")

    def run_all_experiments(self, initial_infected_counts: List[int], n_runs: int = 10, max_steps: int = 5000):
        """Run experiments for all specified initial infected counts."""
        print("=" * 70)
        print(f"Running Initial Infected Variation Experiment")
        print(f"Grid: {self.width}x{self.height}")
        print(f"Initial infected counts to test: {initial_infected_counts}")
        print(f"Runs per count: {n_runs}")
        print("=" * 70)

        for count in initial_infected_counts:
            self.run_ca_for_initial_count(count, n_runs=n_runs, max_steps=max_steps)

        print("\n" + "=" * 70)
        print("Running ODE solutions...")
        print("=" * 70)

        for count in initial_infected_counts:
            self.run_ode_for_initial_count(count, dt=0.1)

    def plot_comparison_grid(self, initial_infected_counts: List[int], save_path: Optional[str] = None):
        """Create a 2-row plot grid:
           - Row 1: CA results (mean ± std) for each initial infected count
           - Row 2: ODE results for each initial infected count
        """
        if not self.ca_results or not self.ode_results:
            raise ValueError("Run experiments first (run_all_experiments).")

        n_cols = len(initial_infected_counts)
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))

        # Ensure axes is always 2D
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        for col_idx, count in enumerate(initial_infected_counts):
            if count not in self.ca_results or count not in self.ode_results:
                print(f"Skipping initial_infected={count} (missing data)")
                continue

            ca_avg = self.ca_results[count]['avg']
            ca_std = self.ca_results[count]['std']
            ode_data = self.ode_results[count]

            # --- CA subplot (top row) ---
            ax_ca = axes[0, col_idx]

            # CA mean and std
            ax_ca.plot(ca_avg['timestep'], ca_avg['S_frac'], label='CA mean S', linestyle='-', linewidth=2, color='blue')
            ax_ca.fill_between(ca_avg['timestep'],
                              ca_avg['S_frac'] - ca_std['S_frac'],
                              ca_avg['S_frac'] + ca_std['S_frac'],
                              alpha=0.2, color='blue')

            ax_ca.plot(ca_avg['timestep'], ca_avg['I_frac'], label='CA mean I', linestyle='-', linewidth=2, color='red')
            ax_ca.fill_between(ca_avg['timestep'],
                              ca_avg['I_frac'] - ca_std['I_frac'],
                              ca_avg['I_frac'] + ca_std['I_frac'],
                              alpha=0.2, color='red')

            ax_ca.plot(ca_avg['timestep'], ca_avg['R_frac'], label='CA mean R', linestyle='-', linewidth=2, color='green')
            ax_ca.fill_between(ca_avg['timestep'],
                              ca_avg['R_frac'] - ca_std['R_frac'],
                              ca_avg['R_frac'] + ca_std['R_frac'],
                              alpha=0.2, color='green')

            ax_ca.set_title(f'CA: Initial Infected = {count}', fontsize=11, fontweight='bold')
            ax_ca.set_xlabel('Timestep')
            ax_ca.set_ylabel('Fraction')
            ax_ca.set_ylim(0, 1)
            ax_ca.legend(loc='best', fontsize='small')
            ax_ca.grid(alpha=0.3)

            # --- ODE subplot (bottom row) ---
            ax_ode = axes[1, col_idx]

            ca_times = ca_avg['timestep'].values
            ode_S_interp = np.interp(ca_times, ode_data['time'], ode_data['S'])
            ode_I_interp = np.interp(ca_times, ode_data['time'], ode_data['I'])
            ode_R_interp = np.interp(ca_times, ode_data['time'], ode_data['R'])

            ax_ode.plot(ca_times, ode_S_interp, label='ODE S', linestyle='-', linewidth=2, color='blue')
            ax_ode.plot(ca_times, ode_I_interp, label='ODE I', linestyle='-', linewidth=2, color='red')
            ax_ode.plot(ca_times, ode_R_interp, label='ODE R', linestyle='-', linewidth=2, color='green')

            ax_ode.set_title(f'ODE: Initial Infected = {count}', fontsize=11, fontweight='bold')
            ax_ode.set_xlabel('Timestep')
            ax_ode.set_ylabel('Fraction')
            ax_ode.set_ylim(0, 1)
            ax_ode.legend(loc='best', fontsize='small')
            ax_ode.grid(alpha=0.3)

        # Add parameter info at bottom
        param_text = f'Grid: {self.width}x{self.height} | Infection prob: {self.infection_prob:.3f} | Recovery prob: {self.recovery_prob:.3f} | Waning prob: {self.waning_prob:.3f} | Mixing rate: {self.mixing_rate:.3f}'
        fig.text(0.5, 0.02, param_text, fontsize=8, family='monospace', ha='center')

        plt.tight_layout(rect=[0, 0.04, 1, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        plt.show()

    def print_summary(self, initial_infected_counts: List[int]):
        """Print a summary of results."""
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"Grid: {self.width}x{self.height}")
        print(f"Infection prob: {self.infection_prob}")
        print(f"Recovery prob: {self.recovery_prob}")
        print(f"Waning prob: {self.waning_prob}")
        print(f"Mixing rate: {self.mixing_rate}")
        print(f"\nInitial infected counts tested: {initial_infected_counts}")

        for count in initial_infected_counts:
            if count in self.ca_results:
                n_runs = len(self.ca_results[count]['runs'])
                max_timestep = self.ca_results[count]['avg']['timestep'].max()
                print(f"\nInitial infected = {count}:")
                print(f"  CA: {n_runs} runs, max timestep: {max_timestep}")
                if count in self.ode_results:
                    max_ode_time = self.ode_results[count]['time'].max()
                    print(f"  ODE: max time: {max_ode_time:.2f}")

        print("=" * 70 + "\n")


def main():
    # Configuration
    width = 50
    height = 50
    infection_prob = 0.08
    recovery_prob = 0.1
    waning_prob = 0.002
    delta_t = 1.0
    mixing_rate = 0.0

    # Test different initial infected counts
    initial_infected_counts = [50, 100, 500, 1000]
    n_runs_per_count = 5  # Number of CA simulations for each initial infected count
    max_steps = 700

    # Run experiment
    experiment = InitialInfectedVariationExperiment(
        width=width,
        height=height,
        infection_prob=infection_prob,
        recovery_prob=recovery_prob,
        waning_prob=waning_prob,
        delta_t=delta_t,
        mixing_rate=mixing_rate
    )

    # Run all experiments
    experiment.run_all_experiments(
        initial_infected_counts=initial_infected_counts,
        n_runs=n_runs_per_count,
        max_steps=max_steps
    )

    # Print summary
    experiment.print_summary(initial_infected_counts)

    # Create and display comparison plots
    experiment.plot_comparison_grid(
        initial_infected_counts=initial_infected_counts,
        save_path="initial_infected_variation_comparison.png"
    )

    print("\nAll done. Files saved:")
    for count in initial_infected_counts:
        print(f" - sirs_ca_initial_{count}.csv")
        print(f" - sirs_ca_avg_initial_{count}.csv")
        print(f" - sirs_ode_initial_{count}.csv")
    print(" - initial_infected_variation_comparison.png")


if __name__ == "__main__":
    main()
