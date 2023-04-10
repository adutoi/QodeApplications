from pytoon        import *
from pytoon.macros import *

tiger = "#F47920"
tiger_cross = composite()
tiger_cross += line(begin=(-0.5,-0.5),end=(0.5,0.5),color=tiger,weight=1/3.)
tiger_cross += line(begin=(0.5,-0.5),end=(-0.5,0.5),color=tiger,weight=1/3.)
tiger_cross = tiger_cross(scale=3)

textX = text(xheight=20)

def to_mEh(nrg):  return 1000*nrg



from data            import be2_fci
from data.xr_100     import be2_bfcorefix
from data.xr_common6 import be2 as be2_common6

from data                 import be3_fci
from data                 import be3_from_xrcc_paper
from data.xr_common6short import be3
from data.xr_common6short import be3_xrfci
from data.xr_common6short import be3_xrfci_updated

dimer = plot2d(ranges=((2.99,10.01),(-7e-4,1e-4)), area=(300,200)).drawFrame()

dimer.plotData(be2_fci.data,       connectors=(black,3), markers=False)
dimer.plotData(be2_bfcorefix.data, connectors=orange,    markers=False)
dimer.plotData(be2_common6.data,   connectors=blue,      markers=False)

#dimer.plotData(be3_fci.data,             connectors=(black,3), markers=False)
#dimer.plotData(be3_from_xrcc_paper.data, connectors=red,       markers=False)
#dimer.plotData(be3.data,                 connectors=blue,       markers=False)
#dimer.plotData(be3_xrfci.data,           connectors=green,       markers=False)
#dimer.plotData(be3_xrfci_updated.data,           connectors=line(color=green,style=dashed),       markers=False)



dimer.setXtics(periodicity=1,   textform=textX("{:.0f}",alignment=(center,top),  translate=(0,-10)))
dimer.setYtics(periodicity=2e-4,textform=(textX("{:.1f}",alignment=(right,center),translate=(-10,-0)),to_mEh))
dimer.pdf("for_poster_be2",texlabels="texlabels/poster/text*.svg")
