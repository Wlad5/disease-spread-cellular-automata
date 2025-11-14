"""
Multi-Parameter Sensitivity Analysis for SIRS CA
================================================
Default behavior: Waning probability and Mixing rate ALWAYS increase
Then vary EITHER infection probability OR recovery probability

This script runs two separate experiments:
1. Vary infection_prob while waning and mixing increase
2. Vary recovery_prob while waning and mixing increase

Each combination generates CA and ODE simulations for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import importlib.util
from scipy.interpolate import interp1d

# ==========================================================
# Dynamic imports of CA and ODE modules
# ==========================================================
ca_path = os.path.join(os.path.dirname(__file__), "SIRS.py")
ode_path = os.path.join(os.path.dirname(__file__), "SIRS_ODE_SOLVER.py")

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

SIRS_CA = import_module_from_path("SIRS_CA", ca_path)
SIRS_ODE = import_module_from_path("SIRS_ODE", ode_path)

# ==========================================================
# Default parameters (base values)
# ==========================================================
base_params = {
    'infection_prob'        : 0.08,
    'recovery_prob'         : 0.1,
    'waning_prob'           : 0.002,
    'k'                     : 8,
    'delta_t'               : 1.0,
    't_max'                 : 200,
    'dt'                    : 1.0,
    'ode_dt'                : 0.1,
    'width'                 : 50,
    'height'                : 50,
    'cell_size'             : 4,
    'initial_infected_count': 10,
    'mixing_rate'           : 0.00,
    'num_simulations'       : 5  # Number of simulations per parameter combination
}

# ==========================================================
# ALWAYS CHANGING: Waning and Mixing (increase together)
# ==========================================================
n_steps = 5  # Number of steps for waning/mixing increase

waning_prob_range = np.linspace(0.0, 1, n_steps)
mixing_rate_range = np.linspace(0.0, 1, n_steps)

# ==========================================================
# VARY SEPARATELY: Infection probability or Recovery probability
# ==========================================================
infection_prob_range = np.linspace(0.00, 1, n_steps)
recovery_prob_range = np.linspace(0.00, 1, n_steps)

# ==========================================================
# Helper functions
# ==========================================================
def compute_initial_infected_fraction(width, height, initial_infected_count):
    """Compute the true fraction of infected population for ODE."""
    total_cells = width * height
    return initial_infected_count / total_cells

def run_ca_simulation(infection_prob, recovery_prob, waning_prob, mixing_rate,
                      t_max, width, height, cell_size, initial_infected_count):
    """Run CA simulation and return history."""
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
    sim.grid.infect_random(n=initial_infected_count)
    sim.running = False
    
    for _ in range(t_max):
        sim.grid.update()
    
    history = sim.get_history()
    return history

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

def calculate_norm(ca_mean, ode_states, ca_timesteps, ode_time):
    """Calculate L2 norm between CA mean and ODE for S, I, R."""
    interp_ode_S = interp1d(ode_time, ode_states[:, 0], kind='linear', fill_value="extrapolate")
    interp_ode_I = interp1d(ode_time, ode_states[:, 1], kind='linear', fill_value="extrapolate")
    interp_ode_R = interp1d(ode_time, ode_states[:, 2], kind='linear', fill_value="extrapolate")
    
    norm_S = np.sqrt(np.sum((ca_mean['S_frac'].values - interp_ode_S(ca_timesteps)) ** 2))
    norm_I = np.sqrt(np.sum((ca_mean['I_frac'].values - interp_ode_I(ca_timesteps)) ** 2))
    norm_R = np.sqrt(np.sum((ca_mean['R_frac'].values - interp_ode_R(ca_timesteps)) ** 2))
    
    return {'S': norm_S, 'I': norm_I, 'R': norm_R}

# ==========================================================
# Experiment 1: Vary Infection Probability
# (Waning & Mixing increase together at each step)
# ==========================================================
def infection_prob_sensitivity_experiment():
    print("\n" + "="*70)
    print("EXPERIMENT 1: Varying Infection Probability")
    print("(Waning probability and Mixing rate INCREASE TOGETHER at each step)")
    print("="*70)
    
    total_cells = base_params['width'] * base_params['height']
    base_frac = compute_initial_infected_fraction(
        base_params['width'], base_params['height'], base_params['initial_infected_count']
    )
    
    print(f"Grid: {base_params['width']}x{base_params['height']} ({total_cells} cells)")
    print(f"Initial infected count: {base_params['initial_infected_count']}")
    print(f"Initial infected fraction (for ODE): {base_frac:.5f}\n")
    
    # Storage: results indexed by (waning_mixing_step, infection_idx)
    # At each waning_mixing_step, BOTH waning and mixing increase together
    ca_results = {}  # (wm_idx, i_idx) -> (mean_df, std_df)
    ode_results = {}  # (wm_idx, i_idx) -> (time, states)
    norms = {}  # (wm_idx, i_idx) -> {'S': float, 'I': float, 'R': float}
    
    total_combos = len(waning_prob_range) * len(infection_prob_range)
    combo_count = 0
    
    for wm_idx, (waning_prob, mixing_rate) in enumerate(zip(waning_prob_range, mixing_rate_range)):
            for i_idx, infection_prob in enumerate(infection_prob_range):
                combo_count += 1
                print(f"[{combo_count}/{total_combos}] waning={waning_prob:.5f}, "
                      f"mixing={mixing_rate:.3f}, infection={infection_prob:.3f}")
                
                # Run multiple CA simulations
                ca_histories = []
                for sim_num in range(base_params['num_simulations']):
                    history = run_ca_simulation(
                        infection_prob=infection_prob,
                        recovery_prob=base_params['recovery_prob'],
                        waning_prob=waning_prob,
                        mixing_rate=mixing_rate,
                        t_max=base_params['t_max'],
                        width=base_params['width'],
                        height=base_params['height'],
                        cell_size=base_params['cell_size'],
                        initial_infected_count=base_params['initial_infected_count']
                    )
                    df = pd.DataFrame({
                        'timestep': history['timestep'],
                        'S_frac': history['S_frac'],
                        'I_frac': history['I_frac'],
                        'R_frac': history['R_frac']
                    })
                    ca_histories.append(df)
                
                # Compute mean and std of CA runs
                combined = pd.concat(ca_histories, ignore_index=True)
                ca_mean = combined.groupby('timestep')[['S_frac', 'I_frac', 'R_frac']].mean().reset_index()
                ca_std = combined.groupby('timestep')[['S_frac', 'I_frac', 'R_frac']].std().reset_index()
                
                ca_results[(wm_idx, i_idx)] = (ca_mean, ca_std)
                
                # Run ODE simulation
                ode_time, ode_states = run_ode_simulation(
                    infection_prob=infection_prob,
                    recovery_prob=base_params['recovery_prob'],
                    waning_prob=waning_prob,
                    k=base_params['k'],
                    delta_t=base_params['delta_t'],
                    initial_infected_frac=base_frac,
                    t_max=base_params['t_max'],
                    ode_dt=base_params['ode_dt']
                )
                ode_results[(wm_idx, i_idx)] = (ode_time, ode_states)
                
                # Calculate norm
                ode_time_normalized = np.linspace(0, len(ode_states) * base_params['ode_dt'], len(ode_states))
                norm = calculate_norm(ca_mean, ode_states, ca_mean['timestep'].values, ode_time_normalized)
                norms[(wm_idx, i_idx)] = norm
    
    plot_infection_sensitivity(infection_prob_range, waning_prob_range, mixing_rate_range, 
                               norms, ca_results, ode_results)
    save_infection_results(infection_prob_range, waning_prob_range, mixing_rate_range, norms)
    
    return norms

# ==========================================================
# Experiment 2: Vary Recovery Probability
# (Waning & Mixing increase together at each step)
# ==========================================================
def recovery_prob_sensitivity_experiment():
    print("\n" + "="*70)
    print("EXPERIMENT 2: Varying Recovery Probability")
    print("(Waning probability and Mixing rate INCREASE TOGETHER at each step)")
    print("="*70)
    
    total_cells = base_params['width'] * base_params['height']
    base_frac = compute_initial_infected_fraction(
        base_params['width'], base_params['height'], base_params['initial_infected_count']
    )
    
    print(f"Grid: {base_params['width']}x{base_params['height']} ({total_cells} cells)")
    print(f"Initial infected count: {base_params['initial_infected_count']}")
    print(f"Initial infected fraction (for ODE): {base_frac:.5f}\n")
    
    # Storage: results indexed by (waning_mixing_step, recovery_idx)
    # At each waning_mixing_step, BOTH waning and mixing increase together
    ca_results = {}
    ode_results = {}
    norms = {}
    
    total_combos = len(waning_prob_range) * len(recovery_prob_range)
    combo_count = 0
    
    for wm_idx, (waning_prob, mixing_rate) in enumerate(zip(waning_prob_range, mixing_rate_range)):
        for r_idx, recovery_prob in enumerate(recovery_prob_range):
            combo_count += 1
            print(f"[{combo_count}/{total_combos}] waning={waning_prob:.5f}, "
                  f"mixing={mixing_rate:.3f}, recovery={recovery_prob:.3f}")
            
            # Run multiple CA simulations
            ca_histories = []
            for sim_num in range(base_params['num_simulations']):
                history = run_ca_simulation(
                    infection_prob=base_params['infection_prob'],
                    recovery_prob=recovery_prob,
                    waning_prob=waning_prob,
                    mixing_rate=mixing_rate,
                    t_max=base_params['t_max'],
                    width=base_params['width'],
                    height=base_params['height'],
                    cell_size=base_params['cell_size'],
                    initial_infected_count=base_params['initial_infected_count']
                )
                df = pd.DataFrame({
                    'timestep': history['timestep'],
                    'S_frac': history['S_frac'],
                    'I_frac': history['I_frac'],
                    'R_frac': history['R_frac']
                })
                ca_histories.append(df)
            
            # Compute mean and std of CA runs
            combined = pd.concat(ca_histories, ignore_index=True)
            ca_mean = combined.groupby('timestep')[['S_frac', 'I_frac', 'R_frac']].mean().reset_index()
            ca_std = combined.groupby('timestep')[['S_frac', 'I_frac', 'R_frac']].std().reset_index()
            
            ca_results[(wm_idx, r_idx)] = (ca_mean, ca_std)
            
            # Run ODE simulation
            ode_time, ode_states = run_ode_simulation(
                infection_prob=base_params['infection_prob'],
                recovery_prob=recovery_prob,
                waning_prob=waning_prob,
                k=base_params['k'],
                delta_t=base_params['delta_t'],
                initial_infected_frac=base_frac,
                t_max=base_params['t_max'],
                ode_dt=base_params['ode_dt']
            )
            ode_results[(wm_idx, r_idx)] = (ode_time, ode_states)
            
            # Calculate norm
            ode_time_normalized = np.linspace(0, len(ode_states) * base_params['ode_dt'], len(ode_states))
            norm = calculate_norm(ca_mean, ode_states, ca_mean['timestep'].values, ode_time_normalized)
            norms[(wm_idx, r_idx)] = norm
    
    
    plot_recovery_sensitivity(recovery_prob_range, waning_prob_range, mixing_rate_range,
                             norms, ca_results, ode_results)
    save_recovery_results(recovery_prob_range, waning_prob_range, mixing_rate_range, norms)
    
    return norms

# ==========================================================
# Plotting Functions - Infection Probability
# ==========================================================
def plot_infection_sensitivity(infection_range, waning_range, mixing_range, norms, ca_results, ode_results):
    """Plot norms for infection probability sensitivity."""
    print("\nGenerating infection probability sensitivity plots...")
    
    # Create one plot for each waning/mixing step
    for wm_idx, (waning, mixing) in enumerate(zip(waning_range, mixing_range)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Extract norms for this waning/mixing combination
        norm_S_list = []
        norm_I_list = []
        norm_R_list = []
        
        for i_idx in range(len(infection_range)):
            norm = norms.get((wm_idx, i_idx), {'S': np.nan, 'I': np.nan, 'R': np.nan})
            norm_S_list.append(norm['S'])
            norm_I_list.append(norm['I'])
            norm_R_list.append(norm['R'])
        
        # Plot norms vs infection probability
        axes[0].plot(infection_range, norm_S_list, 'b-o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Infection Probability', fontsize=11)
        axes[0].set_ylabel('L2 Norm (S)', fontsize=11)
        axes[0].set_title('Susceptible', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(infection_range, norm_I_list, 'r-o', linewidth=2, markersize=8)
        axes[1].set_xlabel('Infection Probability', fontsize=11)
        axes[1].set_ylabel('L2 Norm (I)', fontsize=11)
        axes[1].set_title('Infected', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(infection_range, norm_R_list, 'g-o', linewidth=2, markersize=8)
        axes[2].set_xlabel('Infection Probability', fontsize=11)
        axes[2].set_ylabel('L2 Norm (R)', fontsize=11)
        axes[2].set_title('Recovered', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(f'Infection Sensitivity: Waning={waning:.5f}, Mixing={mixing:.3f}',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'infection_sens_step{wm_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print("Saved infection sensitivity plots.")

def save_infection_results(infection_range, waning_range, mixing_range, norms):
    """Save infection sensitivity results to CSV."""
    rows = []
    for wm_idx, (waning, mixing) in enumerate(zip(waning_range, mixing_range)):
        for i_idx, infection in enumerate(infection_range):
            norm = norms.get((wm_idx, i_idx), {'S': np.nan, 'I': np.nan, 'R': np.nan})
            rows.append({
                'waning_prob': waning,
                'mixing_rate': mixing,
                'infection_prob': infection,
                'norm_S': norm['S'],
                'norm_I': norm['I'],
                'norm_R': norm['R']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv('infection_sensitivity_results.csv', index=False)
    print("Saved infection_sensitivity_results.csv")

# ==========================================================
# Plotting Functions - Recovery Probability
# ==========================================================
def plot_recovery_sensitivity(recovery_range, waning_range, mixing_range, norms, ca_results, ode_results):
    """Plot norms for recovery probability sensitivity."""
    print("\nGenerating recovery probability sensitivity plots...")
    
    # Create one plot for each waning/mixing step
    for wm_idx, (waning, mixing) in enumerate(zip(waning_range, mixing_range)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Extract norms for this waning/mixing combination
        norm_S_list = []
        norm_I_list = []
        norm_R_list = []
        
        for r_idx in range(len(recovery_range)):
            norm = norms.get((wm_idx, r_idx), {'S': np.nan, 'I': np.nan, 'R': np.nan})
            norm_S_list.append(norm['S'])
            norm_I_list.append(norm['I'])
            norm_R_list.append(norm['R'])
        
        # Plot norms vs recovery probability
        axes[0].plot(recovery_range, norm_S_list, 'b-o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Recovery Probability', fontsize=11)
        axes[0].set_ylabel('L2 Norm (S)', fontsize=11)
        axes[0].set_title('Susceptible', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(recovery_range, norm_I_list, 'r-o', linewidth=2, markersize=8)
        axes[1].set_xlabel('Recovery Probability', fontsize=11)
        axes[1].set_ylabel('L2 Norm (I)', fontsize=11)
        axes[1].set_title('Infected', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(recovery_range, norm_R_list, 'g-o', linewidth=2, markersize=8)
        axes[2].set_xlabel('Recovery Probability', fontsize=11)
        axes[2].set_ylabel('L2 Norm (R)', fontsize=11)
        axes[2].set_title('Recovered', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(f'Recovery Sensitivity: Waning={waning:.5f}, Mixing={mixing:.3f}',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'recovery_sens_step{wm_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print("Saved recovery sensitivity plots.")

def save_recovery_results(recovery_range, waning_range, mixing_range, norms):
    """Save recovery sensitivity results to CSV."""
    rows = []
    for wm_idx, (waning, mixing) in enumerate(zip(waning_range, mixing_range)):
        for r_idx, recovery in enumerate(recovery_range):
            norm = norms.get((wm_idx, r_idx), {'S': np.nan, 'I': np.nan, 'R': np.nan})
            rows.append({
                'waning_prob': waning,
                'mixing_rate': mixing,
                'recovery_prob': recovery,
                'norm_S': norm['S'],
                'norm_I': norm['I'],
                'norm_R': norm['R']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv('recovery_sensitivity_results.csv', index=False)
    print("Saved recovery_sensitivity_results.csv")

# ==========================================================
# Main entry point
# ==========================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-PARAMETER SENSITIVITY ANALYSIS FOR SIRS CELLULAR AUTOMATA")
    print("="*70)
    print(f"\nWaning Probability Range: {waning_prob_range}")
    print(f"Mixing Rate Range: {mixing_rate_range}")
    print(f"\nInfection Probability Range: {infection_prob_range}")
    print(f"Recovery Probability Range: {recovery_prob_range}")
    print(f"\nRunning {base_params['num_simulations']} CA simulations per parameter combination")
    
    # Run both experiments
    infection_norms = infection_prob_sensitivity_experiment()
    recovery_norms = recovery_prob_sensitivity_experiment()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print("\nGenerated files:")
    print("  - infection_sensitivity_results.csv")
    print("  - recovery_sensitivity_results.csv")
    print("  - infection_sens_w*_m*.png (multiple plots)")
    print("  - recovery_sens_w*_m*.png (multiple plots)")
