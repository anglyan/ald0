import numpy as np

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
    CF = CostF(ald)

    import matplotlib.pyplot as pt

    t = np.arange(1,8,0.01)
    y = [CF(ti) for ti in t]
    pt.plot(t, y)
    pt.show()



