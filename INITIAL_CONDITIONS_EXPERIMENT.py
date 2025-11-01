"""
Experiment: Vary Initial Conditions in SIRS Cellular Automata
Runs several simulations with the same grid size and parameters, but different initial infected cell configurations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SIRS import Grid
import pygame

# Fixed parameters
WIDTH = 100
HEIGHT = 100
INFECTION_PROB = 0.08
RECOVERY_PROB = 0.1
WANING_PROB = 0.002
DELTA_T = 1.0
MIXING_RATE = 0.00
STEPS = 300
INITIAL_INFECTED_COUNT = 10

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

for label, init_func in initial_conditions:
    grid = Grid(WIDTH, HEIGHT, INFECTION_PROB, RECOVERY_PROB, WANING_PROB, DELTA_T, MIXING_RATE)
    init_func(grid, INITIAL_INFECTED_COUNT)
    cell_size = 6
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * cell_size, HEIGHT * cell_size))
    pygame.display.set_caption(f"SIRS CA - {label}")
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
    results.append(df)
    df.to_csv(f'sirs_initial_{label.replace(" ", "_").lower()}.csv', index=False)
    print(f"Saved results for {label}")
    print(f"Finished simulation for {label}. Close the window to continue.")

# Combine and plot
combined = pd.concat(results, ignore_index=True)
plt.figure(figsize=(12, 8))
for label, _ in initial_conditions:
    subset = combined[combined['initial_condition'] == label]
    plt.plot(subset['timestep'], subset['I_frac'], label=f'I: {label}')
plt.xlabel('Time step')
plt.ylabel('Infected Fraction')
plt.title('SIRS CA: Effect of Initial Conditions')
plt.legend()
plt.tight_layout()
plt.savefig('sirs_initial_conditions_comparison.png')
plt.show()
print("Saved combined plot as sirs_initial_conditions_comparison.png")
