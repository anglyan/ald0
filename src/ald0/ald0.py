"""
ald0

Basic OD ALD model considering the interaction of two precursors with
a growing surface.

All units are in SI unless specifically mentioned:

  - M: molecular mass, atomic mass units
  - T: temperature, Kelvin
  - u: flow velocity, m/s
  - p: precursor pressure, Pa
  - L: radios of reactor height, m
  - sitearea: m^2
  - tp: time to purge: s
  - ald dose and purge times: s

"""

import numpy as np
from scipy.integrate import solve_ivp

kb = 1.38e-23
amu = 1.660e-27


def vth(M, T):
    return np.sqrt(8*kb*T/(np.pi*amu*M))


class ALD0:
    """Implement a 0D ALD process

    Implement a simple 0D model of an ALD process, providing the evolution
    of coverage and thickness as a function of ALD cycles.

    """

    def __init__(self, T, sitearea, chem1, chem2, noise=0.0):
        """
        Parameters
        ----------
        T : float
            temperature in K
        sitearea : float
            area of a surface site, in m^2
        chem1, chem2 : tuples
            tuples containing the chemical paramters of
            the two precursors Each is a five element tuple:

                (p, M, beta, tp, dm)

            where p is the precursor pressure in Pa, M
            is the molecular mass in atomic mass units,
            beta is the sticking probability, tp is
            a characteristic time for precursor evacuation in s,
            and dm is the mass change
            for that particular half-cycle (ng/cm2).
        """

        self.chem1 = chem1
        self.chem2 = chem2

        v1 = vth(chem1.M, T)
        v2 = vth(chem2.M, T)

        self.a01 = sitearea*chem1.beta0*0.25*v1*chem1.p0/(kb*T)
        self.a02 = sitearea*chem2.beta0*0.25*v2*chem2.p0/(kb*T)

        self.tp1 = chem1.tp
        self.tp2 = chem2.tp
        self.dm1 = chem1.dm
        self.dm2 = chem2.dm

        self.c1 = 0
        self.x2 = 0
        self.glist = []

    def run(self, t1, t2, t3, t4, c1=0, x2=0, dt=0.001):
        """Run a single ALD cycle"""

        x1 = t1/(t1+self.tp1)
        t_d1, cov_d1, gr_d1, f_d1, x1, x2 = self.solve_dose1(t1, x1, x2, c1)
        t_p1, cov_p1, gr_p1, f_p1, x1, x2 = self.solve_purge(t2, x1, x2, cov_d1[-1])
        x2 = t3/(t3+self.tp2)
        t_d2, cov_d2, gr_d2, f_d2, x1, x2 = self.solve_dose2(t3, x1, x2, cov_p1[-1])
        t_p2, cov_p2, gr_p2, f_p2, x1, x2 = self.solve_purge(t4, x1, x2, cov_d2[-1])
        self.x1 = x1
        self.x2 = x2
        self.c1 = cov_p2[-1]
        t = np.concatenate([t_d1, t1+t_p1[1:], t1+t2+t_d2[1:], 
                            t1+t2+t3+t_p2[1:]])
        gr = np.concatenate([gr_d1, gr_d1[-1]+gr_p1[1:],
                             gr_d1[-1]+gr_p1[-1]+gr_d2[1:], 
                            gr_d1[-1]+gr_p1[-1]+gr_d2[-1]+gr_p2[1:]])
        f2 = np.concatenate([f_d1, f_d1[-1]+f_p1[1:],
                             f_d1[-1]+f_p1[-1]+f_d2[1:], 
                            f_d1[-1]+f_p1[-1]+f_d2[-1]+f_p2[1:]])
        cov = np.concatenate([cov_d1, cov_p1[1:], cov_d2[1:], 
                            cov_p2[1:]])
        return t, gr, self.dm1*gr + self.dm2*f2, cov


    def _pick_teval(self, t, dt):
        t_eval = np.arange(0, t, dt)
        t_eval = np.concatenate([t_eval, [t]])
        return t_eval

    def solve_dose1(self, t1, x1=1, x2=0, c1=0, dt=0.001): 
        self.x1 = x1
        self.x2 = x2
        if t1 > 0:
            while t1 < dt:
                dt /= 10
            t_eval = self._pick_teval(t1, dt)
            if t_eval[-1] > t1:
                t_eval *= t1/t_eval[-1]
            sol = solve_ivp(self._dose1_f, [0,t1], [c1,0,0], method='BDF',
                t_eval=t_eval, jac=self._dose1_jac)
            return sol.t, sol.y[0], sol.y[1], sol.y[2],\
                self.x1, self.x2*np.exp(-t1/self.tp2)
        else:
            return np.array([0]), np.array([c1]), np.array([0]), np.array([0]), \
                self.x1, self.x2
    
    def _dose1_f(self, t, cov):
        gain = self.a01*self.x1*(1-cov[0])
        loss = self.a02*cov[0]*self.x2*np.exp(-t/self.tp2)
        return np.array([gain-loss, gain, loss])
    
    def _dose1_jac(self, t, cov):
        gj = -self.a01*self.x1
        lj = self.a02*self.x2*np.exp(-t/self.tp2)
        return np.array([
            [gj - lj,0,0],
            [gj,0,0],
            [lj,0,0]])

    def solve_dose2(self, t2, x1=0, x2=0, c1=1, dt=0.001):
        self.x1 = x1
        self.x2 = x2
        if t2 > 0:
            while t2 < dt:
                dt /= 10
            t_eval = self._pick_teval(t2, dt)
            sol = solve_ivp(self._dose2_f, [0,t2], [c1,0,0], method='BDF',
                t_eval=t_eval, jac=self._dose2_jac)
            return sol.t, sol.y[0], sol.y[1], sol.y[2], x1*np.exp(-t2/self.tp1), x2    
        else:
            return np.array([0]), np.array([c1]), np.array([0]), np.array([0]), \
                self.x1, self.x2

    def _dose2_f(self, t, cov):
        gain = self.a01*self.x1*(1-cov[0])*np.exp(-t/self.tp1)
        loss = self.a02*cov[0]
        return np.array([gain-loss, gain, loss])

    def _dose2_jac(self, t, cov):
        gain_jac = -self.a01*self.x1*np.exp(-t/self.tp1)
        loss_jac = self.a02
        return np.array([
            [gain_jac-loss_jac,0, 0],
            [gain_jac, 0,0],
            [loss_jac, 0, 0]])

    def solve_purge(self, tp, x1=0, x2=0, c1=1, dt=0.001):
        self.x1 = x1
        self.x2 = x2
        if tp > 0:
            while tp <= dt:
                dt /= 10
            t_eval = self._pick_teval(tp, dt)
            sol = solve_ivp(self._purge_f, [0,tp], [c1,0,0], method='BDF',
                t_eval=t_eval, jac=self._purge_jac)
            return sol.t, sol.y[0], sol.y[1], sol.y[2], x1*np.exp(-tp/self.tp1), \
                x2*np.exp(-tp/self.tp2)
        else:
            return np.array([0]), np.array([c1]), np.array([0]), np.array([0]), \
                self.x1, self.x2

            
    def _purge_f(self, t, cov):
        gain = self.a01*self.x1*(1-cov[0])*np.exp(-t/self.tp1)
        loss = self.a02*cov[0]*self.x2*np.exp(-t/self.tp2)
        return np.array([gain-loss, gain, loss])

    def _purge_jac(self, t, cov):
        gain_jac = -self.a01*self.x1*np.exp(-t/self.tp1)
        loss_jac = self.a02*self.x2*np.exp(-t/self.tp2)
        return np.array([
            [gain_jac-loss_jac,0,0],
            [gain_jac, 0,0],
            [loss_jac, 0, 0]])


    def cycle(self, t1, t2, t3, t4, N=10, new_growth=True):

        growthrates = []
        if new_growth:
            self.c1 = 0
            self.x2 = 0
        for i in range(N):
            _, gr, _, _ = self.run(t1, t2, t3, t4, c1=self.c1, x2=self.x2)
            growthrates.append(gr[-1])
        self.glist.extend(growthrates)
        return growthrates

