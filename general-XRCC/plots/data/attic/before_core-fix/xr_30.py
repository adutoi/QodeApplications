from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0 .out:Total Excitonic CCSD Energy =  -29.2217809599
3.1 .out:Total Excitonic CCSD Energy =  -29.2218304304
3.2 .out:Total Excitonic CCSD Energy =  -29.2219638424
3.3 .out:Total Excitonic CCSD Energy =  -29.2221709654
3.4 .out:Total Excitonic CCSD Energy =  -29.222432873
3.5 .out:Total Excitonic CCSD Energy =  -29.2227288135
3.6 .out:Total Excitonic CCSD Energy =  -29.2230398477
3.7 .out:Total Excitonic CCSD Energy =  -29.2233504578
3.8 .out:Total Excitonic CCSD Energy =  -29.2236489805
3.9 .out:Total Excitonic CCSD Energy =  -29.223927404
4.0 .out:Total Excitonic CCSD Energy =  -29.2241808602
4.1 .out:Total Excitonic CCSD Energy =  -29.2244070027
4.2 .out:Total Excitonic CCSD Energy =  -29.224605383
4.3 .out:Total Excitonic CCSD Energy =  -29.2247768868
4.4 .out:Total Excitonic CCSD Energy =  -29.224923261
4.5 .out:Total Excitonic CCSD Energy =  -29.2250467403
4.6 .out:Total Excitonic CCSD Energy =  -29.2251497711
4.7 .out:Total Excitonic CCSD Energy =  -29.2252348202
4.8 .out:Total Excitonic CCSD Energy =  -29.2253042521
4.9 .out:Total Excitonic CCSD Energy =  -29.2253602597
5.0 .out:Total Excitonic CCSD Energy =  -29.2254048317
5.1 .out:Total Excitonic CCSD Energy =  -29.2254397446
5.2 .out:Total Excitonic CCSD Energy =  -29.2254665691
5.3 .out:Total Excitonic CCSD Energy =  -29.2254866835
5.4 .out:Total Excitonic CCSD Energy =  -29.2255012898
5.5 .out:Total Excitonic CCSD Energy =  -29.2255114299
5.6 .out:Total Excitonic CCSD Energy =  -29.2255180011
5.7 .out:Total Excitonic CCSD Energy =  -29.2255217695
5.8 .out:Total Excitonic CCSD Energy =  -29.2255233826
5.9 .out:Total Excitonic CCSD Energy =  -29.2255233814
6.0 .out:Total Excitonic CCSD Energy =  -29.2255222109
6.1 .out:Total Excitonic CCSD Energy =  -29.2255202316
6.2 .out:Total Excitonic CCSD Energy =  -29.2255177298
6.3 .out:Total Excitonic CCSD Energy =  -29.2255149278
6.4 .out:Total Excitonic CCSD Energy =  -29.2255119936
6.5 .out:Total Excitonic CCSD Energy =  -29.2255090505
6.6 .out:Total Excitonic CCSD Energy =  -29.2255061849
6.7 .out:Total Excitonic CCSD Energy =  -29.2255034543
6.8 .out:Total Excitonic CCSD Energy =  -29.2255008937
6.9 .out:Total Excitonic CCSD Energy =  -29.2254985211
7.0 .out:Total Excitonic CCSD Energy =  -29.2254963423
7.1 .out:Total Excitonic CCSD Energy =  -29.2254943543
7.2 .out:Total Excitonic CCSD Energy =  -29.2254925487
7.3 .out:Total Excitonic CCSD Energy =  -29.2254909136
7.4 .out:Total Excitonic CCSD Energy =  -29.2254894354
7.5 .out:Total Excitonic CCSD Energy =  -29.2254880998
7.6 .out:Total Excitonic CCSD Energy =  -29.2254868928
7.7 .out:Total Excitonic CCSD Energy =  -29.2254858011
7.8 .out:Total Excitonic CCSD Energy =  -29.2254848124
7.9 .out:Total Excitonic CCSD Energy =  -29.2254839156
""")
