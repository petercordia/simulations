import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json

MAX_STEPS = 1000
SPEED = 20  # steps per second
DEFAULT_X = 0
DEFAULT_K = np.random.randint(0, 2**32)
DEFAULT_SIGMA = 1.0
DEFAULT_BETA = 0.01

class RandomWalkApp:
    def __init__(self, master, x=DEFAULT_X, k=DEFAULT_K, sigma=DEFAULT_SIGMA, beta=DEFAULT_BETA):
        self.master = master
        self.master.title("Random Walk Simulator")

        self.x_buffer = [x]
        self.k_buffer = [k]
        self.sigma_buffer = [sigma]
        self.beta_buffer = [beta]

        self.paused = False

        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.create_controls()
        self.update_walk()

    @property
    def sigma(self):
        return self.sigma_buffer[-1]

    @sigma.setter
    def sigma(self, value):
        self.sigma_buffer[-1] = value

    @property
    def beta(self):
        return self.beta_buffer[-1]

    @beta.setter
    def beta(self, value):
        self.beta_buffer[-1] = value

    def create_controls(self):
        controls_frame = ttk.Frame(self.master)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.create_parameter_control(controls_frame, "Sigma", "sigma", 0.1)
        self.create_parameter_control(controls_frame, "Beta", "beta", 0.01)

        ttk.Button(controls_frame, text="Pause/Resume", command=self.toggle_pause).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Save State", command=self.save_state).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Load State", command=self.load_state).pack(side=tk.LEFT)

    def create_parameter_control(self, parent, label, attr, step):
        frame = ttk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(frame, text=label).pack()
        
        var = tk.StringVar(value=str(getattr(self, attr)))
        entry = ttk.Entry(frame, textvariable=var, width=10)
        entry.pack(side=tk.LEFT)
        entry.bind('<Return>', lambda event, a=attr, v=var: self.update_parameter(a, v))

        def update(delta):
            try:
                value = float(var.get()) + delta
                var.set(f"{value:.4f}")
                setattr(self, attr, value)
            except ValueError:
                pass

        ttk.Button(frame, text="-", command=lambda: update(-step), width=2).pack(side=tk.LEFT)
        ttk.Button(frame, text="+", command=lambda: update(step), width=2).pack(side=tk.LEFT)

    def update_parameter(self, attr, var):
        try:
            value = float(var.get())
            setattr(self, attr, value)
        except ValueError:
            pass

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused:
            self.update_walk()

    def update_walk(self):
        if self.paused:
            return

        self.sigma_buffer.append(self.sigma)
        self.beta_buffer.append(self.beta)

        new_key = np.random.randint(0, 2**32)
        self.k_buffer.append(new_key)

        rng = np.random.default_rng(new_key)
        step = rng.normal(0, self.sigma) - self.beta * self.x_buffer[-1]
        new_x = self.x_buffer[-1] + step
        self.x_buffer.append(new_x)

        if len(self.x_buffer) > MAX_STEPS:
            self.x_buffer = self.x_buffer[-MAX_STEPS:]
            self.k_buffer = self.k_buffer[-MAX_STEPS:]
            self.sigma_buffer = self.sigma_buffer[-MAX_STEPS:]
            self.beta_buffer = self.beta_buffer[-MAX_STEPS:]

        self.ax.clear()
        self.ax.plot(self.x_buffer)
        self.ax.set_title("Random Walk")
        self.canvas.draw()

        self.master.after(int(1000 / SPEED), self.update_walk)

    def save_state(self):
        state = {
            "x_buffer": self.x_buffer,
            "k_buffer": self.k_buffer,
            "sigma_buffer": self.sigma_buffer,
            "beta_buffer": self.beta_buffer,
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(state, f)

    def load_state(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                state = json.load(f)
            self.x_buffer = state["x_buffer"]
            self.k_buffer = state["k_buffer"]
            self.sigma_buffer = state["sigma_buffer"]
            self.beta_buffer = state["beta_buffer"]
            self.update_walk()

def main():
    root = tk.Tk()
    app = RandomWalkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
