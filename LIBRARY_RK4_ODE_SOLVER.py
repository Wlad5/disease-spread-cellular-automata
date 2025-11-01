import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Optional
import csv

class SIRS_ODE:
    def __init__(self, beta: float, alpha: float, gamma: float):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

    def derivatives(self, t, state):
        S, I, R = state
        dS_dt = -self.beta * S * I + self.gamma * R
        dI_dt = self.beta * S * I - self.alpha * I
        dR_dt = self.alpha * I - self.gamma * R
        return [dS_dt, dI_dt, dR_dt]

    def solve(self, initial_state: np.ndarray, t_max: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        t_eval = np.arange(0, t_max + dt, dt)
        sol = solve_ivp(
            fun=self.derivatives,
            t_span=(0, t_max),
            y0=initial_state,
            t_eval=t_eval,
            method='RK45'  # adaptive Runge-Kutta 4(5)
        )
        # Ensure fractions remain [0,1] and normalize
        states = np.clip(sol.y.T, 0, 1)
        states /= states.sum(axis=1, keepdims=True)
        return sol.t, states

    def get_equilibrium(self) -> Tuple[float, float, float]:
        R0 = self.beta / self.alpha
        if R0 <= 1:
            return (1.0, 0.0, 0.0)
        I_eq = (self.gamma * (R0 - 1)) / (self.beta * (self.alpha + self.gamma))
        S_eq = self.alpha / self.beta
        R_eq = 1 - S_eq - I_eq
        return (S_eq, I_eq, R_eq)

    def get_basic_reproduction_number(self) -> float:
        return self.beta / self.alpha


def solve_sirs_from_ca_params(
    infection_prob: float,
    recovery_prob: float,
    waning_prob: float,
    k: int = 8,
    delta_t: float = 1.0,
    initial_infected: float = 0.01,
    t_max: float = 500.0,
    dt: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, dict]:
    # Map CA parameters to ODE parameters
    beta = (infection_prob * k) / delta_t
    alpha = recovery_prob / delta_t
    gamma = waning_prob / delta_t

    model = SIRS_ODE(beta, alpha, gamma)

    # Initial conditions
    I0 = initial_infected
    S0 = 1.0 - I0
    R0 = 0.0
    initial_state = np.array([S0, I0, R0])

    # Solve using library RK4
    time_points, states = model.solve(initial_state, t_max, dt)

    params = {
        'beta': beta,
        'alpha': alpha,
        'gamma': gamma,
        'R0': model.get_basic_reproduction_number(),
        'equilibrium': model.get_equilibrium(),
        'infection_prob': infection_prob,
        'recovery_prob': recovery_prob,
        'waning_prob': waning_prob,
        'k': k,
        'delta_t': delta_t
    }

    return time_points, states, params


def save_to_csv(time_points: np.ndarray, 
                states: np.ndarray, 
                filename: str = "sirs_ode_solution.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'S', 'I', 'R'])
        for t, (S, I, R) in zip(time_points, states):
            writer.writerow([t, S, I, R])
    print(f"ODE solution saved to {filename}")


def plot_sirs_solution(time_points: np.ndarray,
                       states: np.ndarray,
                       params: Optional[dict] = None,
                       save_path: Optional[str] = None):
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, states[:, 0], 'b-', linewidth=2, label='Susceptible (S)')
    plt.plot(time_points, states[:, 1], 'r-', linewidth=2, label='Infected (I)')
    plt.plot(time_points, states[:, 2], 'g-', linewidth=2, label='Recovered (R)')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Fraction of Population', fontsize=12)
    plt.title('SIRS Model - ODE Solution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, time_points[-1])
    plt.ylim(0, 1)

    if params:
        info_text = f"β={params['beta']:.3f}, α={params['alpha']:.3f}, γ={params['gamma']:.3f}, R₀={params['R0']:.2f}"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("SIRS ODE Solver with Library RK4")
    print("="*70)

    time_points, states, params = solve_sirs_from_ca_params(
        infection_prob=0.08,
        recovery_prob=0.1,
        waning_prob=0.002,
        k=8,
        delta_t=1.0,
        initial_infected=0.10,
        t_max=500.0,
        dt=0.1
    )

    print(f"\nODE Parameters:")
    print(f"  β (transmission rate) = {params['beta']:.4f}")
    print(f"  α (recovery rate)     = {params['alpha']:.4f}")
    print(f"  γ (waning rate)       = {params['gamma']:.4f}")
    print(f"  R₀                    = {params['R0']:.4f}")

    S_eq, I_eq, R_eq = params['equilibrium']
    print(f"\nEndemic Equilibrium:")
    print(f"  S* = {S_eq:.4f}")
    print(f"  I* = {I_eq:.4f}")
    print(f"  R* = {R_eq:.4f}")

    save_to_csv(time_points, states, "sirs_ode_solution.csv")
    plot_sirs_solution(time_points, states, params, save_path="sirs_ode_plot_library_RK4.png")

    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70)
