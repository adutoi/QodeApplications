from pytoon        import *
from pytoon.macros import *

tiger = "#F47920"
tiger_cross = composite()
tiger_cross += line(begin=(-0.5,-0.5),end=(0.5,0.5),color=tiger,weight=1/3.)
tiger_cross += line(begin=(0.5,-0.5),end=(-0.5,0.5),color=tiger,weight=1/3.)
tiger_cross = tiger_cross(scale=3)

textX = text(xheight=15)

def to_mEh(nrg):  return 1000*nrg



from data.xr_custom_small import be2_all
from data.xr_custom_small import be3_all
from data.xr_custom_small import be2_1pm
from data.xr_custom_small import be3_1pm
from data import be2_fci_small as be2_fci
from data import be3_fci_small as be3_fci

dimer = plot2d(ranges=((2.99,7.99),(-1e-3,1e-2)), area=(300,200)).drawFrame()

dimer.plotData(be2_fci.data, connectors=(black,3), markers=False)
dimer.plotData(be3_fci.data, connectors=(black,3), markers=False)
dimer.plotData(be2_all.data, connectors=blue,      markers=False)
dimer.plotData(be3_all.data, connectors=blue,      markers=False)
dimer.plotData(be2_1pm.data, connectors=red,       markers=False)
dimer.plotData(be3_1pm.data, connectors=red,       markers=False)

dimer.setXtics(periodicity=1,   textform=textX("{:.0f}",alignment=(center,top),  translate=(0,-10)))
dimer.setYtics(periodicity=2e-3,textform=(textX("{:.1f}",alignment=(right,center),translate=(-10,-0)),to_mEh))
dimer.pdf("PE2_small",texlabels="texlabels/PE2_small/text*.svg")
