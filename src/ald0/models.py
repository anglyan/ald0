from .ald0 import ALD0

aldmodels = {

    'TMA200' :((26.66, 72, 1e-3, 0.1, 1.0), (26.66, 18, 1e-4, 0.1, 0.0), 22.5e-20, 473),
    'TTIP200' : ((0.665, 284, 1e-4,0.1, 1.0),  (26.66, 18, 1e-4, 0.1, 0.0), 117e-20, 473),
    'WF6' : ((6.665, 297, 0.2, 0.1, 1.0), (26.66, 62, 0.05, 0.1, 0), 3.6e-20, 473),
    'TMA100' : ((26.66,72,1e-4,3, 1.0), (26.66, 18, 1e-5, 10, 0), 25.1e-20, 373)
}

def aldprocess(s, noise=0):
    chem1, chem2, s0, T = aldmodels[s]
    return ALD0(T, s0, chem1, chem2, noise)
