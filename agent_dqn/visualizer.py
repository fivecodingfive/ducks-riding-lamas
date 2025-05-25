import matplotlib.pyplot as plt
import numpy as np

class GridVisualizer:
    def __init__(self, grid_size=(5, 5)):
        self.grid_size = grid_size
        self.step = 0

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.ones(self.grid_size), cmap='gray', vmin=0, vmax=1)
        self.texts = [[self.ax.text(j, i, '', ha='center', va='center', fontsize=16, color='black')
                       for j in range(self.grid_size[1])] for i in range(self.grid_size[0])]
        self.ax.set_xticks(np.arange(self.grid_size[1]))
        self.ax.set_yticks(np.arange(self.grid_size[0]))
        self.ax.grid(True)

    def update(self, agent_loc, target_loc, item_locs, reward):
        grid = np.full(self.grid_size, '.', dtype=str)
        ax, ay = agent_loc
        tx, ty = target_loc
        grid[ax, ay] = 'A'
        grid[tx, ty] = 'T'

        for ix, iy in item_locs:
            if grid[ix, iy] == '.':
                grid[ix, iy] = 'I'

        self.ax.set_title(f"Step {self.step} \n Reward:{reward}")
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.texts[i][j].set_text(grid[i, j])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.step += 1
        plt.pause(0.2)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
        print("Visualizer closed")
