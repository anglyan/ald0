import numpy as np
import pickle
import scipy.optimize


from aldfast import ALDdose, CostF, repeated_optimization
from optimize import optimize

def evaluate_cf(a=1, b=10):
    conds = np.loadtxt("/home/yyardi/projects/ald0/optutils/ald_conds.dat")
    n_conds = conds.shape[0]
    path = "/home/yyardi/projects/ald0/optutils/images/testplot"

    optimized_t_values = []
    real_t_values = []

    for i in range(n_conds):
        gpc = conds[i,0]
        t0 = conds[i,1]

        ald = ALDdose(t0, gpc)
        cf = CostF(ald, a, b)
        

        



        # Do stuff
        abs_min_t, abs_min_C, local_mins = optimize(cf)
        # find the error % by taking the actual t-min and the iterations 

        optimized_t_values.append(abs_min_t.item() if isinstance(abs_min_t, np.ndarray) else abs_min_t)

        # repeated_optimization

    with open("/home/yyardi/projects/ald0/optutils/optimized_t_values.pkl", "wb") as f:
        pickle.dump(optimized_t_values, f)

    return optimized_t_values






if __name__ == "__main__":
    optimized_values = evaluate_cf()
    print("Optimized t-values saved:", optimized_values)

    # with open("/home/yyardi/projects/ald0/optutils/optimized_t_values.pkl", "rb") as f:
    #     optimized_t_values = pickle.load(f)
    # print("Loaded optimized t-values:", optimized_t_values)