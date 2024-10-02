import numpy as np
import matplotlib.pyplot as pt

from ald0.ald0 import ALD0

from ald0.models import aldprocess

ald = aldprocess('TMA200')
print(ald.dm1, ald.dm2)

t, gr, mass, cov = ald.run(0.1,1,0.1,1)

pt.plot(t, gr)
pt.plot(t, mass)
pt.plot(t, cov)
pt.show()

