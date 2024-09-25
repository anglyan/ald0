from ald0 import aldprocess
import numpy as np

# We define an ALD process based on chemistry
ald = aldprocess('TMA200', noise=0)

#An ALD process comprises four times tdose1, tpurge1, tdose2, tpurge2

td1 = np.arange(0, 0.1, 0.001)
gr_l = []

for t in td1:
    # The method cycle in ALD0 computes the total thickness per cycle for
    # N consecutive cycles
    gr = np.mean(ald.cycle(t,5,1,5, N=4))
    gr_l.append(gr)

import matplotlib.pyplot as pt

pt.plot(td1, gr_l)
pt.show()


