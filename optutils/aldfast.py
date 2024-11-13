import numpy as np
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence
from optimize import optimize
from skopt import gp_minimize

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
    cf = CostF(ald)
    t_min, C_min = optimize(cf)

    # Plotting final results
    t_values = np.linspace(1, 10, 100)
    C_values = [cf(t) for t in t_values]
    plt.plot(t_values, C_values, label="Cost Function")
    plt.axvline(t_min, color='r', linestyle='--', label=f"Optimal t={t_min:.2f}")
    plt.xlabel("Time (t)")
    plt.ylabel("Cost")
    plt.title("Cost Function Optimization")
    plt.legend()
    plt.savefig("optimized_cost_function.png", dpi=300)
    plt.clf()
    
    plot_convergence(gp_minimize(lambda x: cf(x[0]), [(0.1, 10)], n_calls=15))
    plt.savefig("convergence.png", dpi=300)

    print(f"Optimal t: {t_min}, Minimum Cost: {C_min}")


