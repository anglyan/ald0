from ald0 import aldprocess
import numpy as np
import matplotlib.pyplot as pt
import os

savepath = "/home/yyardi/projects/ald0/"

compound = "TMA200"
# We define an ALD process based on chemistry
ald = aldprocess(compound, noise=0)

#An ALD process comprises four times tdose1, tpurge1, tdose2, tpurge2

td1 = np.arange(0, 0.1, 0.002)
tp1 = np.arange(0, 0.5, 0.002)
gr_l = []

def dose1():
    
    
    for t in td1:
        # The method cycle in ALD0 computes the total thickness per cycle for
        # N consecutive cycles
        gr = np.mean(ald.cycle(t,0.5,0.1,0.5, N=4))
        gr_l.append(gr)
    pt.title(compound + 'Dose 1', fontsize=14, fontweight='bold', color='blue')
    pt.xlabel('Time') 
    pt.ylabel('Thickness') # angstrom
    pt.plot(td1, gr_l)
    pt.savefig(savepath + compound +'Dose 1.png')

def dose2():
    for t in td1:
        # The method cycle in ALD0 computes the total thickness per cycle for
        # N consecutive cycles
        gr = np.mean(ald.cycle(0.1,0.5,t,0.5, N=4))
        gr_l.append(gr)
    pt.title(compound + 'Dose 2', fontsize=14, fontweight='bold', color='blue')
    pt.xlabel('Time')
    pt.ylabel('Thickness')
    pt.plot(td1, gr_l)
    pt.savefig(savepath + compound +'Dose 2.png')


def purge1():
    for t in tp1:
        # The method cycle in ALD0 computes the total thickness per cycle for
        # N consecutive cycles
        gr = np.mean(ald.cycle(0.1,t,0.1,0.5, N=4))
        gr_l.append(gr)
    pt.title(compound + 'Purge 1', fontsize=14, fontweight='bold', color='blue')
    pt.xlabel('Time')
    pt.ylabel('Thickness')
    pt.plot(tp1, gr_l)
    pt.savefig(savepath + compound +'Purge 1.png')

def purge2():
    for t in tp1:
        # The method cycle in ALD0 computes the total thickness per cycle for
        # N consecutive cycles
        gr = np.mean(ald.cycle(0.1,0.5,0.1,t, N=4))
        gr_l.append(gr)
    pt.title(compound + 'Purge 2', fontsize=14, fontweight='bold', color='blue')
    pt.xlabel('Time')
    pt.ylabel('Thickness')
    pt.plot(tp1, gr_l)
    pt.savefig(savepath + compound +'Purge 2.png')

purge2()


