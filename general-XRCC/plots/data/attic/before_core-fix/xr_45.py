from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0 .out:Total Excitonic CCSD Energy =  -29.2255902972
3.1 .out:Total Excitonic CCSD Energy =  -29.225490929
3.2 .out:Total Excitonic CCSD Energy =  -29.2254380585
3.3 .out:Total Excitonic CCSD Energy =  -29.2254201521
3.4 .out:Total Excitonic CCSD Energy =  -29.2254281106
3.5 .out:Total Excitonic CCSD Energy =  -29.2254545628
3.6 .out:Total Excitonic CCSD Energy =  -29.2254932834
3.7 .out:Total Excitonic CCSD Energy =  -29.2255388778
3.8 .out:Total Excitonic CCSD Energy =  -29.2255867068
3.9 .out:Total Excitonic CCSD Energy =  -29.2256329417
4.0 .out:Total Excitonic CCSD Energy =  -29.2256746429
4.1 .out:Total Excitonic CCSD Energy =  -29.2257097916
4.2 .out:Total Excitonic CCSD Energy =  -29.2257372447
4.3 .out:Total Excitonic CCSD Energy =  -29.2257566228
4.4 .out:Total Excitonic CCSD Energy =  -29.2257681525
4.5 .out:Total Excitonic CCSD Energy =  -29.2257724937
4.6 .out:Total Excitonic CCSD Energy =  -29.2257705761
4.7 .out:Total Excitonic CCSD Energy =  -29.2257634602
4.8 .out:Total Excitonic CCSD Energy =  -29.2257522305
4.9 .out:Total Excitonic CCSD Energy =  -29.2257379212
5.0 .out:Total Excitonic CCSD Energy =  -29.2257214698
5.1 .out:Total Excitonic CCSD Energy =  -29.2257036935
5.2 .out:Total Excitonic CCSD Energy =  -29.2256852808
5.3 .out:Total Excitonic CCSD Energy =  -29.2256667944
5.4 .out:Total Excitonic CCSD Energy =  -29.2256486795
5.5 .out:Total Excitonic CCSD Energy =  -29.2256312753
5.6 .out:Total Excitonic CCSD Energy =  -29.2256148285
5.7 .out:Total Excitonic CCSD Energy =  -29.2255995056
5.8 .out:Total Excitonic CCSD Energy =  -29.2255854061
5.9 .out:Total Excitonic CCSD Energy =  -29.225572574
6.0 .out:Total Excitonic CCSD Energy =  -29.2255610088
6.1 .out:Total Excitonic CCSD Energy =  -29.2255506753
6.2 .out:Total Excitonic CCSD Energy =  -29.2255415127
6.3 .out:Total Excitonic CCSD Energy =  -29.225533442
6.4 .out:Total Excitonic CCSD Energy =  -29.2255263735
6.5 .out:Total Excitonic CCSD Energy =  -29.2255202113
6.6 .out:Total Excitonic CCSD Energy =  -29.2255148589
6.7 .out:Total Excitonic CCSD Energy =  -29.2255102219
6.8 .out:Total Excitonic CCSD Energy =  -29.2255062107
6.9 .out:Total Excitonic CCSD Energy =  -29.2255027426
7.0 .out:Total Excitonic CCSD Energy =  -29.2254997424
7.1 .out:Total Excitonic CCSD Energy =  -29.2254971429
7.2 .out:Total Excitonic CCSD Energy =  -29.2254948852
7.3 .out:Total Excitonic CCSD Energy =  -29.2254929179
7.4 .out:Total Excitonic CCSD Energy =  -29.2254911971
7.5 .out:Total Excitonic CCSD Energy =  -29.2254896854
7.6 .out:Total Excitonic CCSD Energy =  -29.2254883512
7.7 .out:Total Excitonic CCSD Energy =  -29.225487168
7.8 .out:Total Excitonic CCSD Energy =  -29.2254861137
7.9 .out:Total Excitonic CCSD Energy =  -29.2254851699
""")
