import numpy as np
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence
from optimize import optimize
import os

class ALDdose:

    def __init__(self, t0=1, gpc=1, r2=None, f2=0, noise=0.01):
        self.t0 = t0
        self.gpc = gpc
        self.noise = noise
        if r2 is None:
            self.single_path = True
        else:
            self.single_path = False
            self.f2 = f2
            self.t2 = self.t0r2
        
    
    def __call__(self, t):
        noise = 1+self.noise*np.random.normal()
        if self.single_path:
            return self.gpc*(1-np.exp(-t/self.t0))*noise
        else:
            g1 = (1-self.f2)*(1-np.exp(-t/self.t0))
            g2 = self.f2*(1-np.exp(-t/self.t2))
            return self.gpc(g1+g2)*noise


class CostF:

    def __init__(self, ald, a=1, b=10, tp=5):
        self.a = a
        self.b = b
        self.ald = ald
        self.tp = tp
        self.cmax = 10

    def __call__(self, t):
        g = self.ald(t)
        dt = 1.1*t
        gp = self.ald(t+dt)
        ep = abs((gp-g)/(dt*gp))
        if t == 0:
            raise ValueError("Only positive times are allowed")
        return self.a*(t+self.tp)/gp + self.b*ep


if __name__ == '__main__':

    ald = ALDdose()
    cf = CostF(ald) # normalize CostF by picking the five points, running the cf and getting, create lambda func to scale the function outputs
    abs_min_t, abs_min_C, local_mins = optimize(cf)
    
    path = "/home/yyardi/projects/ald0/optutils/images"

    # Plotting cost function and abs min
    t_values = np.linspace(1, 10, 100)
    C_values = [cf(t) for t in t_values]
    plt.plot(t_values, C_values, label="Cost Function")
    plt.axvline(abs_min_t, color='r', linestyle='--', label=f"Optimal t={abs_min_t.item():.2f}")
    plt.xlabel("Time (t)")
    plt.ylabel("Cost")
    plt.title("Cost Function Optimization")
    plt.legend()
    plt.savefig(os.path.join(path, "optimized_cost_function.png"), dpi=300)
    plt.clf()

    # plot local mins
    plt.scatter(*zip(*local_mins), color='red', label="Minimums Plotted", zorder=5)
    plt.xlabel("t-values (iter)")
    plt.ylabel("C-values")
    plt.title("Minimums Plotted")
    plt.savefig(os.path.join(path, "minimums_plotted.png"))

    print(f"Optimal t: {abs_min_t}, Minimum Cost: {abs_min_C}")



