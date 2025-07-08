# import os
# import matplotlib

# # Set backend depending on environment
# if os.environ.get("MPLBACKEND") == "Agg":
#     matplotlib.use("Agg")        # For LRZ / headless
# else:
#     matplotlib.use("Qt5Agg")     # For local interactive use

## This works on Mac/Linux dual system
import os
import matplotlib

preferred = os.environ.get("MPL_BACKEND", "Qt5Agg")
try:
    matplotlib.use(preferred)
except ImportError:
    # fallback hierarchy
    for alt in ("TkAgg", "Agg"):
        try:
            matplotlib.use(alt)
            break
        except ImportError:
            continue
else:
    print(f"Using matplotlib backend: {matplotlib.get_backend()!r}")

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

    def update(self, agent_loc, target_loc, item_locs, block_locs, reward, load):
        grid = np.full(self.grid_size, '.', dtype=str)
        ax, ay = agent_loc
        tx, ty = target_loc
        grid[ax, ay] = f'{load}'
        # color = 'blue' if load == 0 else 'red'
        # grid[ax, ay].set_color(color)
            
        grid[tx, ty] = 'T'

        for ix, iy in item_locs:
            if grid[ix, iy] == '.':
                grid[ix, iy] = 'I'
        for bx, by in block_locs:
            if grid[bx, by] == '.':
                grid[bx, by] = 'B'
                # grid[bx, by].set_color('gray')
            

        self.ax.set_title(f"Step {self.step} \n Reward:{reward}")
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.texts[i][j].set_text(grid[i, j])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.step += 1
        plt.pause(0.1)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
        print("Visualizer closed")
