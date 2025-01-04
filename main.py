import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json

MAX_STEPS = 1000
SPEED = 20  # steps per second

class RandomWalkApp:
    # lekker puh
    
    def __init__(self, master):
        self.master = master
        self.master.title("Random Walk Simulator")

        self.x_buffer = [0]
        self.k_buffer = [np.random.randint(0, 2**32)]
        self.sigma_buffer = [1.0]
        self.beta_buffer = [0.01]

        self.sigma = 1.0
        self.beta = 0.01
        self.paused = False

        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.create_controls()
        self.update_walk()

    def create_controls(self):
        controls_frame = ttk.Frame(self.master)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.create_parameter_control(controls_frame, "Sigma", "sigma", 0.1, 10, 0.1)
        self.create_parameter_control(controls_frame, "Beta", "beta", 0, 1, 0.01)

        ttk.Button(controls_frame, text="Pause/Resume", command=self.toggle_pause).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Save State", command=self.save_state).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Load State", command=self.load_state).pack(side=tk.LEFT)

    def create_parameter_control(self, parent, label, attr, min_val, max_val, step):
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
                value = max(min_val, min(max_val, value))
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

        # Check if sigma or beta has been updated
        if self.sigma != self.sigma_buffer[-1] or self.beta != self.beta_buffer[-1]:
            self.sigma_buffer.append(self.sigma)
            self.beta_buffer.append(self.beta)

        # Generate new RNG key
        new_key = np.random.randint(0, 2**32)
        self.k_buffer.append(new_key)

        # Calculate new x value
        rng = np.random.default_rng(new_key)
        step = rng.normal(0, self.sigma) - self.beta * self.x_buffer[-1]
        new_x = self.x_buffer[-1] + step
        self.x_buffer.append(new_x)

        # Trim buffers if they exceed MAX_STEPS
        if len(self.x_buffer) > MAX_STEPS:
            self.x_buffer = self.x_buffer[-MAX_STEPS:]
            self.k_buffer = self.k_buffer[-MAX_STEPS:]
            self.sigma_buffer = self.sigma_buffer[-MAX_STEPS:]
            self.beta_buffer = self.beta_buffer[-MAX_STEPS:]

        # Update plot
        self.ax.clear()
        self.ax.plot(self.x_buffer)
        self.ax.set_title("Random Walk")
        self.canvas.draw()

        if not self.paused:
            self.master.after(int(1000 / SPEED), self.update_walk)

    def save_state(self):
        state = {
            "x_buffer": self.x_buffer,
            "k_buffer": self.k_buffer,
            "sigma_buffer": self.sigma_buffer,
            "beta_buffer": self.beta_buffer,
            "sigma": self.sigma,
            "beta": self.beta
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
            self.sigma = state["sigma"]
            self.beta = state["beta"]
            self.update_walk()

def main():
    root = tk.Tk()
    app = RandomWalkApp(root)
    root.tk.mainloop()

if __name__ == "__main__":
    main()
