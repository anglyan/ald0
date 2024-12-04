import numpy as np

from aldfast import ALDdose, CostF

def evaluate_cf(a=1, b=10):
    conds = np.loadtxt("ald_conds.dat")
    n_conds = conds.shape[0]

    for i in range(n_conds):
        gpc = conds[i,0]
        t0 = conds[i,1]

        ald = ALDdose(t0, gpc)
        cf = CostF(ald, a, b)

        # Do stuff
        print(cf(1.0))


if __name__ == "__main__":

    evaluate_cf()