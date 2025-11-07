"""
SIRS ODE Solver for Disease Spread Model

Mathematical Model:
    dS/dt = -β*S*I + γ*R
    dI/dt = β*S*I - α*I
    dR/dt = α*I - γ*R
    
Where:
    S = fraction of susceptible individuals
    I = fraction of infected individuals
    R = fraction of recovered individuals
    β = transmission rate (infection rate)
    α = recovery rate
    γ = waning immunity rate (rate at which recovered become susceptible again)
    
Parameter Mapping from CA:
    β = p * k / Δt  (where p = infection_prob, k = number of neighbors, Δt = delta_t)
    α = recovery_prob / Δt
    γ = waning_prob / Δt

RK4 formulas for solving ODEs:
Given: dy/dt = f(y, t), initial value y0 at t0, step size h

k1 = f(y_n, t_n)
k2 = f(y_n + 0.5*h*k1, t_n + 0.5*h)
k3 = f(y_n + 0.5*h*k2, t_n + 0.5*h)
k4 = f(y_n + h*k3, t_n + h)

y_{n+1} = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import csv


class SIRS_ODE:
    def __init__(self, beta: float, alpha: float, gamma: float):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        
    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        S, I, R = state
        
        dS_dt = -self.beta * S * I + self.gamma * R
        dI_dt = self.beta * S * I - self.alpha * I
        dR_dt = self.alpha * I - self.gamma * R
        
        return np.array([dS_dt, dI_dt, dR_dt])
    
    def rk4_step(self, state: np.ndarray, t: float, dt: float) -> np.ndarray:
        k1 = self.derivatives(state, t)
        k2 = self.derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.derivatives(state + dt * k3, t + dt)
        
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Ensure states remain in valid range [0, 1]
        new_state = np.clip(new_state, 0, 1)
        
        # Normalize to ensure S + I + R = 1
        total = np.sum(new_state)
        if total > 0:
            new_state = new_state / total
            
        return new_state
    
    def solve(self, 
              initial_state: np.ndarray,
              t_max: float,
              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int(t_max / dt) + 1
        time_points = np.linspace(0, t_max, n_steps)
        states = np.zeros((n_steps, 3))
        
        # Set initial state
        states[0] = initial_state
        
        # Integrate using RK4
        for i in range(1, n_steps):
            states[i] = self.rk4_step(states[i-1], time_points[i-1], dt)
        
        return time_points, states
    
    def get_equilibrium(self) -> Tuple[float, float, float]:
        # Endemic equilibrium exists if β > α (R0 > 1)
        R0 = self.beta / self.alpha
        
        if R0 <= 1:
            # Disease dies out
            return (1.0, 0.0, 0.0)
        
        # Endemic equilibrium
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
    
    # Create ODE model
    model = SIRS_ODE(beta, alpha, gamma)
    
    # Initial conditions
    I0 = initial_infected
    S0 = 1.0 - I0
    R0 = 0.0
    initial_state = np.array([S0, I0, R0])
    
    # Solve using RK4
    time_points, states = model.solve(initial_state, t_max, dt)
    
    # Return with parameter info
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
    print("SIRS ODE Solver - Example Demonstrations")
    print("="*70)
    
    print("\n1. Solving SIRS ODE using CA parameters...")
    time_points, states, params = solve_sirs_from_ca_params(
        infection_prob=0.08,
        recovery_prob=0.1,
        waning_prob=0.002,
        k=8,
        delta_t=1.0,
        initial_infected=0.001,
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
    
    # Save results
    save_to_csv(time_points, states, "sirs_ode_solution.csv")
    
    # Plot solution
    print("\n2. Plotting ODE solution...")
    plot_sirs_solution(time_points, states, params, save_path="sirs_ode_plot.png")
    
    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70)