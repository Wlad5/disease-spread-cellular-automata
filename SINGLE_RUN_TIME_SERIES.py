"""
Single Run Time Series Comparison for SIRS Model
Runs 1 CA experiment with time series plot showing:
  - All compartments (S, I, R): CA result and ODE curve
  - Three separate plots: each state (S / I / R) with CA and ODE
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
import pygame
from SIRS import Grid
from SIRS_ODE_SOLVER import solve_sirs_from_ca_params


class SingleRunTimeSeriesComparison:
    def __init__(self,
                 width: int = 100,
                 height: int = 100,
                 infection_prob: float = 0.08,
                 recovery_prob: float = 0.1,
                 waning_prob: float = 0.002,
                 delta_t: float = 1.0,
                 mixing_rate: float = 0.00,
                 initial_infected_count: int = 10):
        self.width = width
        self.height = height
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.waning_prob = waning_prob
        self.delta_t = delta_t
        self.mixing_rate = mixing_rate
        self.initial_infected_count = initial_infected_count

        # storage
        self.ca_data: Optional[pd.DataFrame] = None
        self.ode_data: Optional[pd.DataFrame] = None
        self.ode_params: Optional[dict] = None

    def run_ca_once(self, max_steps: int = 500) -> pd.DataFrame:
        """Run a single CA simulation headlessly and return timestep series."""
        pygame.init()

        grid = Grid(self.width, self.height,
                    self.infection_prob, self.recovery_prob,
                    self.waning_prob, self.delta_t,
                    self.mixing_rate)
        grid.infect_random(n=self.initial_infected_count)

        for step in range(max_steps):
            grid.update()

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

    def run_ca_simulation(self, max_steps: int = 500):
        """Run single CA simulation."""
        print(f"Running CA simulation (max_steps={max_steps})...")
        df = self.run_ca_once(max_steps=max_steps)
        self.ca_data = df
        df.to_csv('sirs_ca_single_run.csv', index=False)
        print("Saved sirs_ca_single_run.csv")
        return df

    def run_ode_solution(self, dt: float = 0.1):
        """Run ODE solver mapped from CA parameters."""
        if self.ca_data is None:
            max_t = 500.0
        else:
            max_t = float(self.ca_data['timestep'].max())

        # Calculate initial_infected based on number of initially infected cells
        initial_infected = self.initial_infected_count / (self.width * self.height)
        time_points, states, params = solve_sirs_from_ca_params(
            infection_prob=self.infection_prob,
            recovery_prob=self.recovery_prob,
            waning_prob=self.waning_prob,
            k=8,
            delta_t=self.delta_t,
            initial_infected=initial_infected,
            t_max=max_t,
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
        ode_df.to_csv('sirs_ode_single_run.csv', index=False)
        print("Saved sirs_ode_single_run.csv")
        return ode_df

    def plot_l2_norm_difference(self, save_path: Optional[str] = None):
        """Plot L2 norm of the difference between CA and ODE solutions over time.
        
        L2 norm is calculated as: ||diff|| = sqrt((S_diff)^2 + (I_diff)^2 + (R_diff)^2)
        This quantifies overall model agreement at each timestep.
        """
        if self.ca_data is None:
            raise ValueError("Run CA simulation first (run_ca_simulation).")
        if self.ode_data is None:
            raise ValueError("Run ODE solver first (run_ode_solution).")

        ca = self.ca_data
        ode = self.ode_data
        ca_times = ca['timestep'].values

        # Interpolate ODE to CA timesteps
        ode_S_interp = np.interp(ca_times, ode['time'], ode['S'])
        ode_I_interp = np.interp(ca_times, ode['time'], ode['I'])
        ode_R_interp = np.interp(ca_times, ode['time'], ode['R'])

        # Calculate difference vectors at each timestep
        ca_states = np.column_stack([ca['S_frac'].values, ca['I_frac'].values, ca['R_frac'].values])
        ode_states = np.column_stack([ode_S_interp, ode_I_interp, ode_R_interp])
        differences = ca_states - ode_states

        # Compute L2 norm using numpy.linalg.norm
        l2_norms = np.linalg.norm(differences, axis=1, ord=2)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ca_times, l2_norms, color='purple', lw=2.5, label='L2 Norm of (CA - ODE)')
        ax.fill_between(ca_times, 0, l2_norms, alpha=0.3, color='purple')

        ax.set_title('L2 Norm of Difference between CA and ODE (S, I, R)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('L2 Norm ||CA - ODE||₂', fontsize=12)
        ax.grid(alpha=0.4)
        ax.legend(fontsize=11)

        # Add statistics to the plot
        mean_l2 = np.mean(l2_norms)
        max_l2 = np.max(l2_norms)
        stats_text = f'Mean L2: {mean_l2:.4f} | Max L2: {max_l2:.4f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        param_text = f'Grid: {self.width}x{self.height} | β={self.infection_prob:.3f} | γ={self.recovery_prob:.3f} | ξ={self.waning_prob:.3f} | m={self.mixing_rate:.3f}'
        fig.text(0.5, 0.02, param_text, fontsize=9, family='monospace', ha='center')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved L2 norm figure to {save_path}")

        plt.show()

        return l2_norms

    def plot_comparison(self, save_path: Optional[str] = None):
        """Create a 2x2 figure:
           - upper-left: all compartments (CA vs ODE curves)
           - upper-right: S comparison
           - lower-left: I comparison
           - lower-right: R comparison
        """
        if self.ca_data is None:
            raise ValueError("Run CA simulation first (run_ca_simulation).")
        if self.ode_data is None:
            raise ValueError("Run ODE solver first (run_ode_solution).")

        ca = self.ca_data
        ode = self.ode_data

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_all = axes[0, 0]
        ax_S = axes[0, 1]
        ax_I = axes[1, 0]
        ax_R = axes[1, 1]

        # Interpolate ODE to CA timesteps for overlay
        ca_times = ca['timestep'].values
        ode_S_interp = np.interp(ca_times, ode['time'], ode['S'])
        ode_I_interp = np.interp(ca_times, ode['time'], ode['I'])
        ode_R_interp = np.interp(ca_times, ode['time'], ode['R'])

        # --- Top-left: all compartments together ---
        ax_all.plot(ca['timestep'], ca['S_frac'], label='CA S', linestyle='-', linewidth=2, color='blue')
        ax_all.plot(ca['timestep'], ca['I_frac'], label='CA I', linestyle='-', linewidth=2, color='red')
        ax_all.plot(ca['timestep'], ca['R_frac'], label='CA R', linestyle='-', linewidth=2, color='green')

        ax_all.plot(ca_times, ode_S_interp, '--', label='ODE S', linewidth=1.8, color='blue')
        ax_all.plot(ca_times, ode_I_interp, '--', label='ODE I', linewidth=1.8, color='red')
        ax_all.plot(ca_times, ode_R_interp, '--', label='ODE R', linewidth=1.8, color='green')

        ax_all.set_title('All Compartments: CA vs ODE (dashed)')
        ax_all.set_xlabel('Timestep')
        ax_all.set_ylabel('Fraction')
        ax_all.set_ylim(0, 1)
        ax_all.legend(loc='best', fontsize='small')
        ax_all.grid(alpha=0.3)

        # --- Top-right: S comparison ---
        ax_S.plot(ca['timestep'], ca['S_frac'], color='blue', lw=2.5, label='CA S')
        ax_S.plot(ca_times, ode_S_interp, '--', color='blue', lw=1.8, label='ODE S')
        ax_S.set_title('Susceptible (S): CA vs ODE')
        ax_S.set_xlabel('Timestep')
        ax_S.set_ylabel('S fraction')
        ax_S.set_ylim(0, 1)
        ax_S.legend(fontsize='small')
        ax_S.grid(alpha=0.3)

        # --- Bottom-left: I comparison ---
        ax_I.plot(ca['timestep'], ca['I_frac'], color='red', lw=2.5, label='CA I')
        ax_I.plot(ca_times, ode_I_interp, '--', color='red', lw=1.8, label='ODE I')
        ax_I.set_title('Infected (I): CA vs ODE')
        ax_I.set_xlabel('Timestep')
        ax_I.set_ylabel('I fraction')
        ax_I.set_ylim(0, 1)
        ax_I.legend(fontsize='small')
        ax_I.grid(alpha=0.3)

        # --- Bottom-right: R comparison ---
        ax_R.plot(ca['timestep'], ca['R_frac'], color='green', lw=2.5, label='CA R')
        ax_R.plot(ca_times, ode_R_interp, '--', color='green', lw=1.8, label='ODE R')
        ax_R.set_title('Recovered (R): CA vs ODE')
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
        """Print a short summary of CA and ODE params."""
        if self.ca_data is None or self.ode_params is None:
            print("No summary available. Run simulations and ODE first.")
            return

        print("\n=== SUMMARY ===")
        print(f"Grid: {self.width}x{self.height}")
        print(f"CA timesteps: {len(self.ca_data)}")
        print(f"CA final fractions - S: {self.ca_data['S_frac'].iloc[-1]:.4f}, I: {self.ca_data['I_frac'].iloc[-1]:.4f}, R: {self.ca_data['R_frac'].iloc[-1]:.4f}")
        print("ODE params:")
        for k, v in self.ode_params.items():
            print(f"  {k}: {v}")
        print("================\n")


def main():
    print("=" * 70)
    print("SIRS Model: Single CA and ODE run with Time Series Comparison")
    print("=" * 70)

    width = 200
    height = 200
    infection_prob = 0.08
    recovery_prob = 0.1
    waning_prob = 0.002
    delta_t = 1.0
    mixing_rate = 0.7
    initial_infected_count = 30000
    max_steps = 500

    comparison = SingleRunTimeSeriesComparison(
        width=width,
        height=height,
        infection_prob=infection_prob,
        recovery_prob=recovery_prob,
        waning_prob=waning_prob,
        delta_t=delta_t,
        mixing_rate=mixing_rate,
        initial_infected_count=initial_infected_count
    )

    ca_df = comparison.run_ca_simulation(max_steps=max_steps)

    ode_df = comparison.run_ode_solution(dt=0.1)

    comparison.print_summary()

    comparison.plot_comparison(save_path="sirs_single_run_comparison.png")

    l2_norms = comparison.plot_l2_norm_difference(save_path="sirs_l2_norm_difference.png")

    print("\nAll done. Files saved:")
    print(" - sirs_ca_single_run.csv")
    print(" - sirs_ode_single_run.csv")
    print(" - sirs_single_run_comparison.png")
    print(" - sirs_l2_norm_difference.png")
    print(" - sirs_single_run_comparison.png")


if __name__ == "__main__":
    main()
