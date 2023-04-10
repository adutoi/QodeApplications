from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0 .out:Total Excitonic CCSD Energy =  -29.223776552473176
3.1 .out:Total Excitonic CCSD Energy =  -29.22391731131087
3.2 .out:Total Excitonic CCSD Energy =  -29.224106946349384
3.3 .out:Total Excitonic CCSD Energy =  -29.22431877650776
3.4 .out:Total Excitonic CCSD Energy =  -29.224534991571844
3.5 .out:Total Excitonic CCSD Energy =  -29.224744115664098
3.6 .out:Total Excitonic CCSD Energy =  -29.224938897105595
3.7 .out:Total Excitonic CCSD Energy =  -29.225114857931768
3.8 .out:Total Excitonic CCSD Energy =  -29.225269418735028
3.9 .out:Total Excitonic CCSD Energy =  -29.225401405536655
4.0 .out:Total Excitonic CCSD Energy =  -29.225510756836435
4.1 .out:Total Excitonic CCSD Energy =  -29.22559830745151
4.2 .out:Total Excitonic CCSD Energy =  -29.22566558714861
4.3 .out:Total Excitonic CCSD Energy =  -29.225714616833685
4.4 .out:Total Excitonic CCSD Energy =  -29.22574770938809
4.5 .out:Total Excitonic CCSD Energy =  -29.225767290044605
4.6 .out:Total Excitonic CCSD Energy =  -29.22577574900587
4.7 .out:Total Excitonic CCSD Energy =  -29.22577533264144
4.8 .out:Total Excitonic CCSD Energy =  -29.225768073046154
4.9 .out:Total Excitonic CCSD Energy =  -29.225755751009885
5.0 .out:Total Excitonic CCSD Energy =  -29.22573988503111
5.1 .out:Total Excitonic CCSD Energy =  -29.22572173857982
5.2 .out:Total Excitonic CCSD Energy =  -29.22570233868549
5.3 .out:Total Excitonic CCSD Energy =  -29.22568250043378
5.4 .out:Total Excitonic CCSD Energy =  -29.225662853574953
5.5 .out:Total Excitonic CCSD Energy =  -29.225643868865582
5.6 .out:Total Excitonic CCSD Energy =  -29.225625882843133
5.7 .out:Total Excitonic CCSD Energy =  -29.225609120459055
5.8 .out:Total Excitonic CCSD Energy =  -29.2255937154081
5.9 .out:Total Excitonic CCSD Energy =  -29.22557972820341
6.0 .out:Total Excitonic CCSD Energy =  -29.22556716211305
6.1 .out:Total Excitonic CCSD Energy =  -29.22555597707937
6.2 .out:Total Excitonic CCSD Energy =  -29.225546101725374
6.3 .out:Total Excitonic CCSD Energy =  -29.225537443544322
6.4 .out:Total Excitonic CCSD Energy =  -29.22552989736137
6.5 .out:Total Excitonic CCSD Energy =  -29.225523352177547
6.6 .out:Total Excitonic CCSD Energy =  -29.225517696522015
6.7 .out:Total Excitonic CCSD Energy =  -29.225512822459677
6.8 .out:Total Excitonic CCSD Energy =  -29.225508628422528
6.9 .out:Total Excitonic CCSD Energy =  -29.225505021040355
7.0 .out:Total Excitonic CCSD Energy =  -29.225501916149362
7.1 .out:Total Excitonic CCSD Energy =  -29.22549923915295
7.2 .out:Total Excitonic CCSD Energy =  -29.22549692489203
7.3 .out:Total Excitonic CCSD Energy =  -29.225494917168696
7.4 .out:Total Excitonic CCSD Energy =  -29.225493168042327
7.5 .out:Total Excitonic CCSD Energy =  -29.225491636997106
7.6 .out:Total Excitonic CCSD Energy =  -29.225490290057312
7.7 .out:Total Excitonic CCSD Energy =  -29.225489098907783
7.8 .out:Total Excitonic CCSD Energy =  -29.22548804005909
7.9 .out:Total Excitonic CCSD Energy =  -29.22548709408288
""")
