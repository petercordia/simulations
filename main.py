import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from jax import random, Array
from time import time

MAX_STEPS = 1000

class RandomWalkApp:
    buffer = list[float]
    k_buffer = list[Array]
    sigma_buffer = list[float]
    beta_buffer : list[float]

    def __init__(self, master, x=0, rng_seed=None, sigma=1.0, beta=0.01, speed=20):
        if rng_seed is None:
            rng_seed = int(time())

        self.master = master
        self.master.title("Random Walk Simulator")

        self.x_buffer = [x]
        self.k_buffer = [random.key(rng_seed)]
        self.sigma_buffer = [sigma]
        self.beta_buffer = [beta]
        self.speed = speed

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

    @property
    def x(self):
        return self.x_buffer[-1]

    @property
    def k(self):
        return self.k_buffer[-1]
    
    def create_controls(self):
        controls_frame = ttk.Frame(self.master)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.create_parameter_control(controls_frame, "Sigma", "sigma", 0.1)
        self.create_parameter_control(controls_frame, "Beta", "beta", 0.01)
        self.create_parameter_control(controls_frame, "Speed", "speed", 1)

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

        key, subkey = random.split(self.k)
        self.k_buffer.append(key)

        new_x = float(self.x * (1 - self.beta) + random.normal(subkey) * self.sigma)
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

        self.master.after(int(1000 / self.speed), self.update_walk)

    def save_state(self):
        state = {
            "x_buffer": self.x_buffer,
            "k_buffer": [[int(x) for x in random.key_data(key)] for key in self.k_buffer],
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
            self.k_buffer = [random.wrap_key_data(Array(key_data)) for key_data in state["k_buffer"]]
            self.sigma_buffer = state["sigma_buffer"]
            self.beta_buffer = state["beta_buffer"]
            self.update_walk()

def main():
    root = tk.Tk()
    app = RandomWalkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
