from .ald0 import ALD0
from collections import namedtuple

ALDPrec = namedtuple("ALDPrec", ["p0", "M", "beta0", "tp", "dm"])
ALDChem = namedtuple("ALDChem", ["chem1", "chem2", "s0", "T", "r"])


aldmodels = {

    'TMA200' : ALDChem(
        ALDPrec(26.66, 72, 1e-3, 0.1, 1.0),
        ALDPrec(26.66, 18, 1e-4, 0.1, 0.0),
        22.5e-20, 473, 1.5),
    'TTIP200' : ALDChem(
        ALDPrec(0.665, 284, 1e-4,0.1, 1.0),
        ALDPrec(26.66, 18, 1e-4, 0.1, 0.0),
        117e-20, 473, 2.0),
    'WF6' : ALDChem(
        ALDPrec(6.665, 297, 0.2, 0.1, 1.0),
        ALDPrec(26.66, 62, 0.05, 0.1, 0),
        3.6e-20, 473, 1),
    'TMA100' : ALDChem(
        ALDPrec(26.66,72,1e-4, 3, 1.0),
        ALDPrec(26.66, 18, 1e-5, 10, 0),
        25.1e-20, 373, 1.5)
}

def aldprocess(s, noise=0):
    aldchem = aldmodels[s]
    return ALD0(aldchem.T, aldchem.s0, aldchem.chem1, aldchem.chem2, noise)
