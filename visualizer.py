import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import ast
import csv

# --- Load replay buffer from CSV ---
replay_buffer = []
with open("last_replay_buffer.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        step_data = {
            "agent_loc": ast.literal_eval(row["agent_loc"]),
            "target_loc": ast.literal_eval(row["target_loc"]),
            "item_locs": ast.literal_eval(row["item_locs"]),
            "item_times": ast.literal_eval(row["item_times"]),
            "reward": float(row["reward"]),
            "agent_load": int(row["agent_load"]),
        }
        replay_buffer.append(step_data)

# --- Define GridVisualizer ---
class GridVisualizer:
    def __init__(self, grid_size=(5, 5)):
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.im = self.ax.imshow(np.ones(self.grid_size), cmap='gray', vmin=0, vmax=1)
        self.texts = [
            [self.ax.text(j, i, '', ha='center', va='center', fontsize=16, color='black')
             for j in range(self.grid_size[1])]
            for i in range(self.grid_size[0])
        ]
        self.ax.set_xticks(np.arange(self.grid_size[1]))
        self.ax.set_yticks(np.arange(self.grid_size[0]))
        self.ax.grid(True)

    def update(self, agent_loc, target_loc, item_locs, item_times, reward, agent_load, step=0):
        grid = np.full(self.grid_size, '.', dtype=str)
        ax, ay = agent_loc
        tx, ty = target_loc
        grid[ax, ay] = 'A'
        grid[tx, ty] = 'T'
        # Items mit Restzeit eintragen:
        for (ix, iy), time_left in zip(item_locs, item_times):
            if grid[ix, iy] == '.':
                grid[ix, iy] = str(time_left)
        self.ax.set_title(f"Step {step}\nReward: {reward:.2f}")
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.texts[i][j].set_text(grid[i, j])
                if (i, j) == agent_loc and agent_load == 1:
                    self.texts[i][j].set_color('red')
                else:
                    self.texts[i][j].set_color('black')
        self.fig.canvas.draw_idle()


    def close(self):
        plt.close(self.fig)

# --- Visualization with buttons ---
def visualize_buffer_buttons(replay_buffer):
    visualizer = GridVisualizer()
    step = [0]  # Mutable int to persist in closures

    def show_frame():
        data = replay_buffer[step[0]]
        visualizer.update(
            data["agent_loc"], data["target_loc"], data["item_locs"],
            data["item_times"],   # <--- Hier einfügen!
            data["reward"], data["agent_load"], step[0]
        )


    def on_prev(b):
        if step[0] > 0:
            step[0] -= 1
            show_frame()
    def on_next(b):
        if step[0] < len(replay_buffer) - 1:
            step[0] += 1
            show_frame()

    prev_btn = widgets.Button(description="Previous")
    next_btn = widgets.Button(description="Next")
    label = widgets.Label(value=f"Step: {step[0]}")

    def update_label(*args):
        label.value = f"Step: {step[0]} / {len(replay_buffer) - 1}"

    prev_btn.on_click(lambda b: [on_prev(b), update_label()])
    next_btn.on_click(lambda b: [on_next(b), update_label()])

    ui = widgets.HBox([prev_btn, next_btn, label])
    display(ui)
    show_frame()

visualize_buffer_buttons(replay_buffer)
