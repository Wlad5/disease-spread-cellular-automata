import pygame
import numpy as np
import random

# SIRS states
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

# Colors for visualization
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
            recovery_time=30,
            susceptible_time=50
            ):
        self.width = width
        self.height = height
        self.infection_prob = infection_prob
        self.recovery_time = recovery_time
        self.susceptible_time = susceptible_time
        self.grid = np.array([[Cell() for _ in range(width)] for _ in range(height)])

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
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append(self.grid[ny][nx])
        return neighbors

    def update(self):
        new_grid = np.array([[Cell(cell.state) for cell in row] for row in self.grid])
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.state == SUSCEPTIBLE:
                    neighbors = self.get_neighbors(x, y)
                    if any(n.state == INFECTED for n in neighbors):
                        if random.random() < self.infection_prob:
                            new_grid[y][x].state = INFECTED
                elif cell.state == INFECTED:
                    cell.infection_time += 1
                    if cell.infection_time >= self.recovery_time:
                        new_grid[y][x].state = RECOVERED
                        new_grid[y][x].recovered_time = 0
                    else:
                        new_grid[y][x].infection_time = cell.infection_time
                elif cell.state == RECOVERED:
                    cell.recovered_time += 1
                    if cell.recovered_time >= self.susceptible_time:
                        new_grid[y][x].state = SUSCEPTIBLE
                        new_grid[y][x].infection_time = 0
                        new_grid[y][x].recovered_time = 0
                    else:
                        new_grid[y][x].recovered_time = cell.recovered_time
        self.grid = new_grid

class Simulation:
    def __init__(
            self,
            width=100,
            height=100,
            cell_size=6,
            infection_prob=0.25,
            recovery_time=30,
            susceptible_time=50
            ):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("SIRS Disease Spread Cellular Automata")
        self.grid = Grid(width, height, infection_prob, recovery_time, susceptible_time)
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

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.grid.update()
            self.draw()
            self.clock.tick(60)  # FPS

        pygame.quit()

if __name__ == "__main__":
    sim = Simulation(width=100, height=100, cell_size=6, infection_prob=0.15, recovery_time=7, susceptible_time=90)
    sim.run()