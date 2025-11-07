"""
Time Series Comparison for SIRS Model
Modified to run 15 CA experiments and show 4 plots:
  - All compartments (S, I, R): CA mean ± std and ODE curves
  - Three separate plots: each state (S / I / R) with CA runs, CA mean, and ODE
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List
import pygame
from SIRS import Grid
from SIRS_ODE_SOLVER import solve_sirs_from_ca_params


class TimeSeriesComparison:
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

        # storage
        self.ca_runs: List[pd.DataFrame] = []
        self.ca_combined: Optional[pd.DataFrame] = None
        self.ca_avg: Optional[pd.DataFrame] = None
        self.ca_std: Optional[pd.DataFrame] = None

        self.ode_data: Optional[pd.DataFrame] = None
        self.ode_params: Optional[dict] = None

    def run_ca_once(self, max_steps: int = 5000) -> pd.DataFrame:
        """Run a single CA simulation headlessly (uses a tiny pygame surface) and return timestep series."""
        pygame.init()

        grid = Grid(self.width, self.height,
                    self.infection_prob, self.recovery_prob,
                    self.waning_prob, self.delta_t,
                    self.mixing_rate)
        grid.infect_random(n=10) #/////////////////////////////////////////////////////////////////////////

        for step in range(max_steps):
            grid.update()
            _, I_frac, _ = grid.get_population_fractions()
            # early stop if epidemic died out
            if step > 100 and I_frac < 0.0001:
                # print(f"  CA: died out at step {step}")
                break

        # record final fractions (the code you provided records before updates; we ensure final record)
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

    def run_multiple_ca_simulations(self, n_runs: int = 15, max_steps: int = 5000):
        """Run n_runs independent CA simulations and compute average + std per timestep."""
        print(f"Running {n_runs} CA simulations (max_steps={max_steps})...")
        all_runs = []
        for i in range(n_runs):
            print(f"  ▶️ Run {i+1}/{n_runs}")
            df = self.run_ca_once(max_steps=max_steps)
            df = df.copy()
            df['run'] = i + 1
            all_runs.append(df)

        # Save individual run list and combined
        self.ca_runs = all_runs
        combined = pd.concat(all_runs, ignore_index=True)
        self.ca_combined = combined

        # compute mean and std grouped by timestep (works even if some runs end earlier)
        avg = combined.groupby('timestep')[['S_frac', 'I_frac', 'R_frac']].mean().reset_index()
        std = combined.groupby('timestep')[['S_frac', 'I_frac', 'R_frac']].std().reset_index()

        self.ca_avg = avg
        self.ca_std = std

        # save combined results
        combined.to_csv('sirs_ca_15runs.csv', index=False)
        avg.to_csv('sirs_ca_15runs_avg.csv', index=False)
        std.to_csv('sirs_ca_15runs_std.csv', index=False)
        print("Saved sirs_ca_15runs.csv, sirs_ca_15runs_avg.csv, sirs_ca_15runs_std.csv")
        return combined

    def run_ode_solution(self, dt: float = 0.1):
        """Run ODE solver mapped from CA parameters using t_max equal to max CA timestep."""
        if self.ca_runs:
            max_t = max(df['timestep'].max() for df in self.ca_runs)
            t_max = float(max_t)
        else:
            t_max = 500.0

        initial_infected = 0.001 / (self.width * self.height)  # same initial infected used in CA ///////////////////////////////////////
        time_points, states, params = solve_sirs_from_ca_params(
            infection_prob=self.infection_prob,
            recovery_prob=self.recovery_prob,
            waning_prob=self.waning_prob,
            k=8,
            delta_t=self.delta_t,
            initial_infected=initial_infected,
            t_max=t_max,
            dt=dt
        )

        ode_df = pd.DataFrame({
            'time': time_points,
            'S': states[:, 0],
            'I': states[:, 1],
            'R': states[:, 2]
        })
        self.ode_data = ode_df
        self.ode_params = params
        # save ode result
        ode_df.to_csv('sirs_ode_solution.csv', index=False)
        print("Saved sirs_ode_solution.csv")
        return ode_df

    def plot_four_panels(self, save_path: Optional[str] = None):
        """Create a 2x2 figure:
           - upper-left: all compartments (CA mean ± std vs ODE curves)
           - upper-right: S comparison (all runs, CA mean, ODE)
           - lower-left: I comparison
           - lower-right: R comparison
        """
        if self.ca_avg is None or self.ca_std is None or self.ca_runs is None:
            raise ValueError("Run CA simulations first (run_multiple_ca_simulations).")
        if self.ode_data is None:
            raise ValueError("Run ODE solver first (run_ode_solution).")

        avg = self.ca_avg
        std = self.ca_std
        ode = self.ode_data

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax_all = axes[0, 0]
        ax_S = axes[0, 1]
        ax_I = axes[1, 0]
        ax_R = axes[1, 1]

        # --- Top-left: all compartments together (mean ± std from CA and ODE curves) ---
        # CA mean and std fill for S, I, R
        ax_all.plot(avg['timestep'], avg['S_frac'], label='CA mean S', linestyle='-', linewidth=2, color='blue')
        ax_all.fill_between(avg['timestep'],
                            avg['S_frac'] - std['S_frac'],
                            avg['S_frac'] + std['S_frac'],
                            alpha=0.2)
        ax_all.plot(avg['timestep'], avg['I_frac'], label='CA mean I', linestyle='-', linewidth=2, color='red')
        ax_all.fill_between(avg['timestep'],
                            avg['I_frac'] - std['I_frac'],
                            avg['I_frac'] + std['I_frac'],
                            alpha=0.2)
        ax_all.plot(avg['timestep'], avg['R_frac'], label='CA mean R', linestyle='-', linewidth=2, color='green')
        ax_all.fill_between(avg['timestep'],
                            avg['R_frac'] - std['R_frac'],
                            avg['R_frac'] + std['R_frac'],
                            alpha=0.2)

        # ODE curves (interpolate ODE times to CA timesteps for nicer overlay)
        ca_times = avg['timestep'].values
        ode_S_interp = np.interp(ca_times, ode['time'], ode['S'])
        ode_I_interp = np.interp(ca_times, ode['time'], ode['I'])
        ode_R_interp = np.interp(ca_times, ode['time'], ode['R'])

        ax_all.plot(ca_times, ode_S_interp, '--', label='ODE S', linewidth=1.8, color='blue')
        ax_all.plot(ca_times, ode_I_interp, '--', label='ODE I', linewidth=1.8, color='red')
        ax_all.plot(ca_times, ode_R_interp, '--', label='ODE R', linewidth=1.8, color='green')

        ax_all.set_title('All Compartments: CA mean ± std vs ODE (dashed)')
        ax_all.set_xlabel('Timestep')
        ax_all.set_ylabel('Fraction')
        ax_all.set_ylim(0, 1)
        ax_all.legend(loc='best', fontsize='small')
        ax_all.grid(alpha=0.3)

        # --- Top-right: S comparisons ---
        for idx, df in enumerate(self.ca_runs):
            ax_S.plot(df['timestep'], df['S_frac'], alpha=0.25, lw=1)
        ax_S.plot(avg['timestep'], avg['S_frac'], color='blue', lw=2.5, label='CA mean S')
        ax_S.plot(ca_times, ode_S_interp, '--', color='blue', lw=1.8, label='ODE S')
        ax_S.set_title('Susceptible (S): CA runs, CA mean, ODE')
        ax_S.set_xlabel('Timestep')
        ax_S.set_ylabel('S fraction')
        ax_S.set_ylim(0, 1)
        ax_S.legend(fontsize='small')
        ax_S.grid(alpha=0.3)

        # --- Bottom-left: I comparisons ---
        for idx, df in enumerate(self.ca_runs):
            ax_I.plot(df['timestep'], df['I_frac'], alpha=0.25, lw=1, color='tab:red')
        ax_I.plot(avg['timestep'], avg['I_frac'], color='tab:red', lw=2.5, label='CA mean I')
        ax_I.plot(ca_times, ode_I_interp, '--', color='tab:red', lw=1.8, label='ODE I')
        ax_I.set_title('Infected (I): CA runs, CA mean, ODE')
        ax_I.set_xlabel('Timestep')
        ax_I.set_ylabel('I fraction')
        ax_I.set_ylim(0, 1)
        ax_I.legend(fontsize='small')
        ax_I.grid(alpha=0.3)

        # --- Bottom-right: R comparisons ---
        for idx, df in enumerate(self.ca_runs):
            ax_R.plot(df['timestep'], df['R_frac'], alpha=0.25, lw=1, color='tab:green')
        ax_R.plot(avg['timestep'], avg['R_frac'], color='tab:green', lw=2.5, label='CA mean R')
        ax_R.plot(ca_times, ode_R_interp, '--', color='tab:green', lw=1.8, label='ODE R')
        ax_R.set_title('Recovered (R): CA runs, CA mean, ODE')
        ax_R.set_xlabel('Timestep')
        ax_R.set_ylabel('R fraction')
        ax_R.set_ylim(0, 1)
        ax_R.legend(fontsize='small')
        ax_R.grid(alpha=0.3)

        param_text = f'Grid: {self.width}x{self.height} | Infection prob: {self.infection_prob:.3f} | Recovery prob: {self.recovery_prob:.3f} | Waning prob: {self.waning_prob:.3f} | Mixing rate: {self.mixing_rate:.3f}'
        fig.text(0.5, 0.02, param_text, fontsize=8, family='monospace', ha='center')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        plt.show()

    def print_summary(self):
        """Print a short summary of CA average and ODE params."""
        if self.ca_avg is None or self.ode_params is None:
            print("No summary available. Run simulations and ODE first.")
            return

        print("\n=== SUMMARY ===")
        print(f"Grid: {self.width}x{self.height}")
        print(f"CA runs: {len(self.ca_runs)}")
        print("ODE params:")
        for k, v in self.ode_params.items():
            print(f"  {k}: {v}")
        print("================\n")


def main():
    print("=" * 70)
    print("SIRS Model: Run 15 CA experiments and compare to ODE")
    print("=" * 70)

    width = 50
    height = 50
    infection_prob = 0.08
    recovery_prob = 0.1
    waning_prob = 0.002
    delta_t = 1.0
    mixing_rate = 0.0
    n_runs = 10
    max_steps = 500

    comparison = TimeSeriesComparison(
        width=width,
        height=height,
        infection_prob=infection_prob,
        recovery_prob=recovery_prob,
        waning_prob=waning_prob,
        delta_t=delta_t,
        mixing_rate=mixing_rate
    )

    combined = comparison.run_multiple_ca_simulations(n_runs=n_runs, max_steps=max_steps)

    ode_df = comparison.run_ode_solution(dt=0.1)

    comparison.print_summary()

    comparison.plot_four_panels(save_path=f"sirs_{n_runs}runs_comparison.png")

    print("\nAll done. Files saved:")
    print(" - sirs_ca_15runs.csv")
    print(" - sirs_ca_15runs_avg.csv")
    print(" - sirs_ca_15runs_std.csv")
    print(" - sirs_ode_solution.csv")
    print(" - sirs_15runs_comparison.png")


if __name__ == "__main__":
    main()