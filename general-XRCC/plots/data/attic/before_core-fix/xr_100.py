from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0 .out:Total Excitonic CCSD Energy =  -29.2225875259
3.1 .out:Total Excitonic CCSD Energy =  -29.2230239616
3.2 .out:Total Excitonic CCSD Energy =  -29.2234364759
3.3 .out:Total Excitonic CCSD Energy =  -29.2238203665
3.4 .out:Total Excitonic CCSD Energy =  -29.2241710476
3.5 .out:Total Excitonic CCSD Energy =  -29.2244851768
3.6 .out:Total Excitonic CCSD Energy =  -29.2247609671
3.7 .out:Total Excitonic CCSD Energy =  -29.2249981382
3.8 .out:Total Excitonic CCSD Energy =  -29.2251977192
3.9 .out:Total Excitonic CCSD Energy =  -29.2253617915
4.0 .out:Total Excitonic CCSD Energy =  -29.2254932136
4.1 .out:Total Excitonic CCSD Energy =  -29.2255953499
4.2 .out:Total Excitonic CCSD Energy =  -29.2256718218
4.3 .out:Total Excitonic CCSD Energy =  -29.2257262965
4.4 .out:Total Excitonic CCSD Energy =  -29.2257623204
4.5 .out:Total Excitonic CCSD Energy =  -29.2257832001
4.6 .out:Total Excitonic CCSD Energy =  -29.2257919286
4.7 .out:Total Excitonic CCSD Energy =  -29.2257911479
4.8 .out:Total Excitonic CCSD Energy =  -29.2257831417
4.9 .out:Total Excitonic CCSD Energy =  -29.225769848
5.0 .out:Total Excitonic CCSD Energy =  -29.2257528833
5.1 .out:Total Excitonic CCSD Energy =  -29.2257335748
5.2 .out:Total Excitonic CCSD Energy =  -29.2257129933
5.3 .out:Total Excitonic CCSD Energy =  -29.2256919864
5.4 .out:Total Excitonic CCSD Energy =  -29.2256712094
5.5 .out:Total Excitonic CCSD Energy =  -29.2256511529
5.6 .out:Total Excitonic CCSD Energy =  -29.225632169
5.7 .out:Total Excitonic CCSD Energy =  -29.2256144932
5.8 .out:Total Excitonic CCSD Energy =  -29.2255982656
5.9 .out:Total Excitonic CCSD Energy =  -29.225583549
6.0 .out:Total Excitonic CCSD Energy =  -29.2255703455
6.1 .out:Total Excitonic CCSD Energy =  -29.225558611
6.2 .out:Total Excitonic CCSD Energy =  -29.225548268
6.3 .out:Total Excitonic CCSD Energy =  -29.2255392165
6.4 .out:Total Excitonic CCSD Energy =  -29.225531343
6.5 .out:Total Excitonic CCSD Energy =  -29.2255245281
6.6 .out:Total Excitonic CCSD Energy =  -29.225518652
6.7 .out:Total Excitonic CCSD Energy =  -29.2255135992
6.8 .out:Total Excitonic CCSD Energy =  -29.225509261
6.9 .out:Total Excitonic CCSD Energy =  -29.225505538
7.0 .out:Total Excitonic CCSD Energy =  -29.2255023405
7.1 .out:Total Excitonic CCSD Energy =  -29.2254995895
7.2 .out:Total Excitonic CCSD Energy =  -29.2254972161
7.3 .out:Total Excitonic CCSD Energy =  -29.2254951609
7.4 .out:Total Excitonic CCSD Energy =  -29.2254933736
7.5 .out:Total Excitonic CCSD Energy =  -29.2254918116
7.6 .out:Total Excitonic CCSD Energy =  -29.2254904394
7.7 .out:Total Excitonic CCSD Energy =  -29.2254892274
7.8 .out:Total Excitonic CCSD Energy =  -29.2254881512
7.9 .out:Total Excitonic CCSD Energy =  -29.2254871905
""")
