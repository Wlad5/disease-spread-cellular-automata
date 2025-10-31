import pygame
import numpy as np
import random
import csv

SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

COLORS = {
    SUSCEPTIBLE: (200, 200, 200),  # light gray
    INFECTED: (200, 0, 0),         # red
    RECOVERED: (0, 150, 0)         # green
}

class Cell:
    def __init__(self, state=SUSCEPTIBLE):
        self.state = state
        self.infection_time = 0
        self.recovered_time = 0
class Grid:
    def __init__(
            self,
            width,
            height,
            infection_prob=0.25,
            recovery_prob=0.05,
            waning_prob=0.02,
            delta_t=1,
            mixing_rate=0.05 # fraction of population to mix per timestep
            ):
        self.width = width
        self.height = height
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.waning_prob = waning_prob
        self.delta_t = delta_t
        self.grid = np.array([[Cell() for _ in range(width)] for _ in range(height)])
        self.mixing_rate = mixing_rate
        # Add tracking for population fractions
        self.history = {
            'S_frac': [],
            'I_frac': [],
            'R_frac': [],
            'timestep': []
        }
        self.current_timestep = 0

    def infect_random(self, n=10):
        for _ in range(n):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.grid[y][x].state = INFECTED

    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                neighbors.append(self.grid[ny][nx])
        return neighbors
    
    def get_population_fractions(self):
        """Calculate current fraction of S, I, R individuals"""
        total = self.width * self.height
        S_count = 0
        I_count = 0
        R_count = 0
        
        for y in range(self.height):
            for x in range(self.width):
                state = self.grid[y][x].state
                if state == SUSCEPTIBLE:
                    S_count += 1
                elif state == INFECTED:
                    I_count += 1
                elif state == RECOVERED:
                    R_count += 1
        
        return S_count / total, I_count / total, R_count / total
    
    def record_fractions(self):
        """Record current population fractions to history"""
        S_frac, I_frac, R_frac = self.get_population_fractions()
        self.history['S_frac'].append(S_frac)
        self.history['I_frac'].append(I_frac)
        self.history['R_frac'].append(R_frac)
        self.history['timestep'].append(self.current_timestep)

    def mix_population(self):
        """Randomly swap the states of a fraction of the population."""
        total_cells = self.width * self.height
        n_swaps = int(self.mixing_rate * total_cells)
        for _ in range(n_swaps):
            x1, y1 = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            x2, y2 = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            # Swap states
            self.grid[y1][x1].state, self.grid[y2][x2].state = self.grid[y2][x2].state, self.grid[y1][x1].state

    def update(self):
        # Record fractions before update
        self.record_fractions()
        # Mix population before disease update
        self.mix_population()
        new_grid = np.array([[Cell(cell.state) for cell in row] for row in self.grid])
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.state == SUSCEPTIBLE:
                    neighbors = self.get_neighbors(x, y)
                    n_infected = sum(1 for n in neighbors if n.state == INFECTED)
                    if n_infected > 0:
                        p = self.infection_prob * self.delta_t
                        infection_probability = 1 - (1 - p) ** n_infected
                        if random.random() < infection_probability:
                            new_grid[y][x].state = INFECTED
                elif cell.state == INFECTED:
                    if random.random() < self.recovery_prob * self.delta_t:
                        new_grid[y][x].state = RECOVERED
                        new_grid[y][x].recovered_time = 0
                elif cell.state == RECOVERED:
                    if random.random() < self.waning_prob * self.delta_t:
                        new_grid[y][x].state = SUSCEPTIBLE
                        new_grid[y][x].infection_time = 0
                        new_grid[y][x].recovered_time = 0
        self.grid = new_grid
        self.current_timestep += 1
class Simulation:
    def __init__(
            self,
            width=100,
            height=100,
            cell_size=6,
            infection_prob=0.25,
            recovery_prob=0.05,
            waning_prob=0.02,
            delta_t=1,
            mixing_rate=0.05
            ):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("SIRS Disease Spread Cellular Automata")
        self.grid = Grid(width, height, infection_prob, recovery_prob, waning_prob, delta_t, mixing_rate)
        self.grid.infect_random(n=8)
        self.clock = pygame.time.Clock()
        self.running = True

    def draw(self):
        for y in range(self.height):
            for x in range(self.width):
                state = self.grid.grid[y][x].state
                color = COLORS[state]
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        pygame.display.flip()
    
    def get_history(self):
        """Get the recorded population fraction history"""
        return self.grid.history
    
    def save_history(self, filename='sirs_population_fractions.csv'):
        """Save population fraction history to CSV file"""
        history = self.grid.history
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'S_frac', 'I_frac', 'R_frac'])
            for i in range(len(history['timestep'])):
                writer.writerow([
                    history['timestep'][i],
                    history['S_frac'][i],
                    history['I_frac'][i],
                    history['R_frac'][i]
                ])
        print(f"History saved to {filename}")

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.grid.update()
            self.draw()
            self.clock.tick(60)

        # Save history when simulation ends
        self.save_history()
        pygame.quit()

if __name__ == "__main__":
    sim = Simulation(
        width=100, 
        height=100, 
        cell_size=6, 
        infection_prob=0.08,
        recovery_prob=0.1,     
        waning_prob=0.002,      
        delta_t=1,
        mixing_rate=0.000
    )
    sim.run()
    
    history = sim.get_history()
    if len(history['timestep']) > 0:
        print(f"\nSimulation Summary:")
        print(f"Total timesteps recorded: {len(history['timestep'])}")
        print(f"Final S fraction: {history['S_frac'][-1]:.4f}")
        print(f"Final I fraction: {history['I_frac'][-1]:.4f}")
        print(f"Final R fraction: {history['R_frac'][-1]:.4f}")