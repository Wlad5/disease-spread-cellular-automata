"""
Experiment: Vary Initial Conditions in SIRS Cellular Automata
Runs several simulations with the same grid size and parameters, but different initial infected cell configurations.
Also varies grid sizes to observe scaling effects.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SIRS import Grid
import pygame
from SIRS_ODE_SOLVER import solve_sirs_from_ca_params
import os

# Create output directories
os.makedirs('initial_conditions_csv', exist_ok=True)
os.makedirs('initial_conditions_images', exist_ok=True)

def compute_l2_norm(ca_data, ode_data):
    """Compute L2 norm (RMSE) between CA and ODE trajectories"""
    return np.sqrt(np.mean((ca_data - ode_data)**2))

# Fixed parameters
WIDTH = 20
HEIGHT = 20
INFECTION_PROB = 0.08
RECOVERY_PROB = 0.1
WANING_PROB = 0.002
DELTA_T = 1.0
MIXING_RATE = 0.00
STEPS = 500
INITIAL_INFECTED_COUNT = 10
NUM_REPLICATES = 2  # Number of simulations to run per initial condition

# Grid sizes to vary
GRID_SIZES = [50, 100, 150, 200, 300]

# Pre-compute ODE solution for comparison
# Initial infected fraction for ODE (matching CA initial condition)
INITIAL_INFECTED_FRACTION = INITIAL_INFECTED_COUNT / (WIDTH * HEIGHT)
ode_time, ode_states, ode_params = solve_sirs_from_ca_params(
    infection_prob=INFECTION_PROB,
    recovery_prob=RECOVERY_PROB,
    waning_prob=WANING_PROB,
    k=8,  # 8 neighbors in CA
    delta_t=DELTA_T,
    initial_infected=INITIAL_INFECTED_FRACTION,
    t_max=STEPS,
    dt=1.0
)

# Different initial condition strategies

# Initial condition strategies
def random_infected(grid, n):
    grid.infect_random(n=n)

def clustered_infected(grid, n):
    cx, cy = WIDTH // 2, HEIGHT // 2
    infected = 0
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if infected < n:
                x = (cx + dx) % WIDTH
                y = (cy + dy) % HEIGHT
                grid.grid[y][x].state = 1
                infected += 1

def corners_infected(grid, n):
    corners = [(0,0), (0,HEIGHT-1), (WIDTH-1,0), (WIDTH-1,HEIGHT-1)]
    infected = 0
    for idx, (cx, cy) in enumerate(corners):
        for dx in range(0, 2):
            for dy in range(0, 2):
                if infected < n:
                    x = (cx + dx) % WIDTH
                    y = (cy + dy) % HEIGHT
                    grid.grid[y][x].state = 1
                    infected += 1
    while infected < n:
        x = (corners[0][0] + infected) % WIDTH
        y = (corners[0][1] + infected) % HEIGHT
        grid.grid[y][x].state = 1
        infected += 1


def all_susceptible(grid, n):
    # All cells are susceptible, then infect n cells
    for y in range(HEIGHT):
        for x in range(WIDTH):
            grid.grid[y][x].state = 0
    grid.infect_random(n=n)

def all_infected(grid, n):
    # All cells are infected
    for y in range(HEIGHT):
        for x in range(WIDTH):
            grid.grid[y][x].state = 1


def all_recovered(grid, n):
    # All cells are recovered, then infect n cells
    for y in range(HEIGHT):
        for x in range(WIDTH):
            grid.grid[y][x].state = 2
    grid.infect_random(n=n)

initial_conditions = [
    ("Random", random_infected),
    ("Clustered Center", clustered_infected),
    ("Corners", corners_infected),
    ("All Susceptible", all_susceptible),
    ("All Infected", all_infected),
    ("All Recovered", all_recovered)
]

results = []

# Dictionary to store L2 norms and statistics for each initial condition across grid sizes
norms_by_condition = {label: {'grid_sizes': [], 'S_norms': [], 'I_norms': [], 'R_norms': [],
                               'S_means': [], 'S_stds': [], 'I_means': [], 'I_stds': [],
                               'R_means': [], 'R_stds': []} 
                      for label, _ in initial_conditions}


def draw_grid(grid, cell_size=6, title=None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * cell_size, HEIGHT * cell_size))
    if title:
        pygame.display.set_caption(title)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            state = grid.grid[y][x].state
            color = (200, 200, 200) if state == 0 else (200, 0, 0) if state == 1 else (0, 150, 0)
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)
    pygame.display.flip()

# Loop over different grid sizes
for grid_size in GRID_SIZES:
    WIDTH = grid_size
    HEIGHT = grid_size
    INITIAL_INFECTED_COUNT = max(1, grid_size // 2)  # Scale infected count with grid size
    
    # Pre-compute ODE solution for this grid size
    INITIAL_INFECTED_FRACTION = INITIAL_INFECTED_COUNT / (WIDTH * HEIGHT)
    ode_time, ode_states, ode_params = solve_sirs_from_ca_params(
        infection_prob=INFECTION_PROB,
        recovery_prob=RECOVERY_PROB,
        waning_prob=WANING_PROB,
        k=8,  # 8 neighbors in CA
        delta_t=DELTA_T,
        initial_infected=INITIAL_INFECTED_FRACTION,
        t_max=STEPS,
        dt=1.0
    )
    
    grid_results = []
    
    for label, init_func in initial_conditions:
        # Store data from all replicates for this initial condition
        replicate_dataframes = []
        replicate_S_means = []
        replicate_I_means = []
        replicate_R_means = []
        replicate_S_stds = []
        replicate_I_stds = []
        replicate_R_stds = []
        replicate_norms = {'S': [], 'I': [], 'R': []}
        
        for rep in range(NUM_REPLICATES):
            grid = Grid(WIDTH, HEIGHT, INFECTION_PROB, RECOVERY_PROB, WANING_PROB, DELTA_T, MIXING_RATE)
            init_func(grid, INITIAL_INFECTED_COUNT)
            cell_size = 6
            pygame.init()
            screen = pygame.display.set_mode((WIDTH * cell_size, HEIGHT * cell_size))
            pygame.display.set_caption(f"SIRS CA - {label} ({WIDTH}x{HEIGHT}) - Replicate {rep+1}/{NUM_REPLICATES}")
            clock = pygame.time.Clock()
            running = True
            step = 0
            while running and step < STEPS:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                grid.update()
                # Draw grid
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        state = grid.grid[y][x].state
                        color = (200, 200, 200) if state == 0 else (200, 0, 0) if state == 1 else (0, 150, 0)
                        rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                        pygame.draw.rect(screen, color, rect)
                        pygame.draw.rect(screen, (50, 50, 50), rect, 1)
                pygame.display.flip()
                clock.tick(60)
                step += 1
            pygame.quit()
            history = grid.history
            df = pd.DataFrame({
                'timestep': history['timestep'],
                'S_frac': history['S_frac'],
                'I_frac': history['I_frac'],
                'R_frac': history['R_frac']
            })
            df['initial_condition'] = label
            df['replicate'] = rep
            replicate_dataframes.append(df)
            
            # Compute L2 norms between CA and ODE solutions
            ca_timesteps = df['timestep'].values
            ode_interp_S = np.interp(ca_timesteps, ode_time, ode_states[:, 0])
            ode_interp_I = np.interp(ca_timesteps, ode_time, ode_states[:, 1])
            ode_interp_R = np.interp(ca_timesteps, ode_time, ode_states[:, 2])
            
            S_norm = compute_l2_norm(df['S_frac'].values, ode_interp_S)
            I_norm = compute_l2_norm(df['I_frac'].values, ode_interp_I)
            R_norm = compute_l2_norm(df['R_frac'].values, ode_interp_R)
            
            replicate_norms['S'].append(S_norm)
            replicate_norms['I'].append(I_norm)
            replicate_norms['R'].append(R_norm)
            
            # Compute mean and standard deviation for this replicate's CA trajectory
            S_mean = np.mean(df['S_frac'].values)
            S_std = np.std(df['S_frac'].values)
            I_mean = np.mean(df['I_frac'].values)
            I_std = np.std(df['I_frac'].values)
            R_mean = np.mean(df['R_frac'].values)
            R_std = np.std(df['R_frac'].values)
            
            replicate_S_means.append(S_mean)
            replicate_S_stds.append(S_std)
            replicate_I_means.append(I_mean)
            replicate_I_stds.append(I_std)
            replicate_R_means.append(R_mean)
            replicate_R_stds.append(R_std)
            
            print(f"  Replicate {rep+1}/{NUM_REPLICATES}: S: mean={S_mean:.4f}, std={S_std:.4f} | I: mean={I_mean:.4f}, std={I_std:.4f} | R: mean={R_mean:.4f}, std={R_std:.4f}")
        
        # Combine all replicates for this initial condition
        combined_replicates = pd.concat(replicate_dataframes, ignore_index=True)
        grid_results.append(combined_replicates)
        results.append(combined_replicates)
        
        # Compute mean and std of the norms and statistics across all replicates
        S_norm_mean = np.mean(replicate_norms['S'])
        S_norm_std = np.std(replicate_norms['S'])
        I_norm_mean = np.mean(replicate_norms['I'])
        I_norm_std = np.std(replicate_norms['I'])
        R_norm_mean = np.mean(replicate_norms['R'])
        R_norm_std = np.std(replicate_norms['R'])
        
        S_mean_mean = np.mean(replicate_S_means)
        S_mean_std = np.std(replicate_S_means)
        I_mean_mean = np.mean(replicate_I_means)
        I_mean_std = np.std(replicate_I_means)
        R_mean_mean = np.mean(replicate_R_means)
        R_mean_std = np.std(replicate_R_stds)
        
        norms_by_condition[label]['grid_sizes'].append(grid_size)
        norms_by_condition[label]['S_norms'].append(S_norm_mean)
        norms_by_condition[label]['I_norms'].append(I_norm_mean)
        norms_by_condition[label]['R_norms'].append(R_norm_mean)
        norms_by_condition[label]['S_means'].append(S_mean_mean)
        norms_by_condition[label]['S_stds'].append(S_mean_std)
        norms_by_condition[label]['I_means'].append(I_mean_mean)
        norms_by_condition[label]['I_stds'].append(I_mean_std)
        norms_by_condition[label]['R_means'].append(R_mean_mean)
        norms_by_condition[label]['R_stds'].append(R_mean_std)
        
        csv_filename = f'initial_conditions_csv/sirs_initial_{label.replace(" ", "_").lower()}_{WIDTH}x{HEIGHT}.csv'
        combined_replicates.to_csv(csv_filename, index=False)
        print(f"Saved results for {label} ({WIDTH}x{HEIGHT})")
        print(f"  Across {NUM_REPLICATES} replicates:")
        print(f"    S: mean={S_mean_mean:.4f} ± {S_mean_std:.4f}")
        print(f"    I: mean={I_mean_mean:.4f} ± {I_mean_std:.4f}")
        print(f"    R: mean={R_mean_mean:.4f} ± {R_mean_std:.4f}")
    
    # Create a figure with 2x3 subplots for this grid size
    combined_grid = pd.concat(grid_results, ignore_index=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Plot each scenario
    for idx, (label, _) in enumerate(initial_conditions):
        subset = combined_grid[combined_grid['initial_condition'] == label]
        ax = axes[idx]
        
        # Compute mean and std for each timestep
        mean_S = subset.groupby('timestep')['S_frac'].mean()
        std_S = subset.groupby('timestep')['S_frac'].std()
        mean_I = subset.groupby('timestep')['I_frac'].mean()
        std_I = subset.groupby('timestep')['I_frac'].std()
        mean_R = subset.groupby('timestep')['R_frac'].mean()
        std_R = subset.groupby('timestep')['R_frac'].std()
        
        timesteps = mean_S.index.values
        
        # Plot shaded areas for mean ± std
        ax.fill_between(timesteps, mean_S - std_S, mean_S + std_S, color='blue', alpha=0.2, label='S (CA mean ± std)')
        ax.fill_between(timesteps, mean_I - std_I, mean_I + std_I, color='red', alpha=0.2, label='I (CA mean ± std)')
        ax.fill_between(timesteps, mean_R - std_R, mean_R + std_R, color='green', alpha=0.2, label='R (CA mean ± std)')
        
        # Plot mean lines
        ax.plot(timesteps, mean_S.values, '-', color='blue', linewidth=2)
        ax.plot(timesteps, mean_I.values, '-', color='red', linewidth=2)
        ax.plot(timesteps, mean_R.values, '-', color='green', linewidth=2)
        
        # Plot ODE curves (dotted lines)
        ax.plot(ode_time, ode_states[:, 0], ':', label='S (ODE)', color='blue', alpha=0.7, linewidth=2)
        ax.plot(ode_time, ode_states[:, 1], ':', label='I (ODE)', color='red', alpha=0.7, linewidth=2)
        ax.plot(ode_time, ode_states[:, 2], ':', label='R (ODE)', color='green', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Time step')
        ax.set_ylabel('Population Fraction')
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Add common title and parameters
    plt.suptitle(f'SIRS CA: Population Dynamics for Different Initial Conditions ({WIDTH}x{HEIGHT} Grid, {NUM_REPLICATES} runs) (CA vs ODE)', y=0.95)
    params_text = f'CA Parameters: Grid={WIDTH}x{HEIGHT}, β={INFECTION_PROB}, γ={RECOVERY_PROB}, ξ={WANING_PROB}, Mix={MIXING_RATE}, Initial Infected={INITIAL_INFECTED_COUNT}, Replicates={NUM_REPLICATES}\n'
    params_text += f'ODE Parameters: β={ode_params["beta"]:.4f}, α={ode_params["alpha"]:.4f}, γ={ode_params["gamma"]:.4f}, R₀={ode_params["R0"]:.2f}'
    plt.figtext(0.5, 0.01, params_text, ha='center', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # Save plot
    plot_filename = f'initial_conditions_images/sirs_initial_conditions_comparison_{WIDTH}x{HEIGHT}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot as {plot_filename}")
    plt.close()

print("All simulations and plots completed!")

# Create L2 norms plot showing how norms vary with grid size
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Plot L2 norms for each compartment (S, I, R)
for idx, (label, _) in enumerate(initial_conditions):
    ax = axes[idx]
    
    grid_sizes = norms_by_condition[label]['grid_sizes']
    S_norms = norms_by_condition[label]['S_norms']
    I_norms = norms_by_condition[label]['I_norms']
    R_norms = norms_by_condition[label]['R_norms']
    
    ax.plot(grid_sizes, S_norms, 'o-', label='S (Susceptible)', color='blue', linewidth=2, markersize=8)
    ax.plot(grid_sizes, I_norms, 's-', label='I (Infected)', color='red', linewidth=2, markersize=8)
    ax.plot(grid_sizes, R_norms, '^-', label='R (Recovered)', color='green', linewidth=2, markersize=8)
    
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('L2 Norm (RMSE)')
    ax.set_title(f'{label}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(GRID_SIZES)

plt.suptitle('L2 Norms (CA vs ODE) for Different Initial Conditions Across Grid Sizes', y=0.995)
params_text = f'CA Parameters: β={INFECTION_PROB}, γ={RECOVERY_PROB}, ξ={WANING_PROB}, Mix={MIXING_RATE}'
plt.figtext(0.5, 0.02, params_text, ha='center', fontsize=9)

plt.tight_layout()
plt.subplots_adjust(top=0.96, bottom=0.08)

# Save L2 norms plot
norms_plot_filename = 'initial_conditions_images/sirs_l2_norms_vs_grid_size.png'
plt.savefig(norms_plot_filename, dpi=150, bbox_inches='tight')
print(f"Saved L2 norms plot as {norms_plot_filename}")
plt.close()

print("L2 norms plots completed!")