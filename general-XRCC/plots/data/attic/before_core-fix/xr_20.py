from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0 .out:Total Excitonic CCSD Energy =  -29.214629699
3.1 .out:Total Excitonic CCSD Energy =  -29.2157029058
3.2 .out:Total Excitonic CCSD Energy =  -29.2167276538
3.3 .out:Total Excitonic CCSD Energy =  -29.2177018239
3.4 .out:Total Excitonic CCSD Energy =  -29.2186181005
3.5 .out:Total Excitonic CCSD Energy =  -29.2194687201
3.6 .out:Total Excitonic CCSD Energy =  -29.220247852
3.7 .out:Total Excitonic CCSD Energy =  -29.2209524801
3.8 .out:Total Excitonic CCSD Energy =  -29.2215824186
3.9 .out:Total Excitonic CCSD Energy =  -29.2221398858
4.0 .out:Total Excitonic CCSD Energy =  -29.2226289027
4.1 .out:Total Excitonic CCSD Energy =  -29.2230546737
4.2 .out:Total Excitonic CCSD Energy =  -29.2234230376
4.3 .out:Total Excitonic CCSD Energy =  -29.2237400263
4.4 .out:Total Excitonic CCSD Energy =  -29.2240115418
4.5 .out:Total Excitonic CCSD Energy =  -29.2242431417
4.6 .out:Total Excitonic CCSD Energy =  -29.2244399171
4.7 .out:Total Excitonic CCSD Energy =  -29.2246064398
4.8 .out:Total Excitonic CCSD Energy =  -29.2247467603
4.9 .out:Total Excitonic CCSD Energy =  -29.2248644371
5.0 .out:Total Excitonic CCSD Energy =  -29.2249625832
5.1 .out:Total Excitonic CCSD Energy =  -29.2250439194
5.2 .out:Total Excitonic CCSD Energy =  -29.2251108268
5.3 .out:Total Excitonic CCSD Energy =  -29.2251653941
5.4 .out:Total Excitonic CCSD Energy =  -29.2252094577
5.5 .out:Total Excitonic CCSD Energy =  -29.2252446338
5.6 .out:Total Excitonic CCSD Energy =  -29.2252723447
5.7 .out:Total Excitonic CCSD Energy =  -29.2252938387
5.8 .out:Total Excitonic CCSD Energy =  -29.2253102061
5.9 .out:Total Excitonic CCSD Energy =  -29.2253223927
6.0 .out:Total Excitonic CCSD Energy =  -29.2253312117
6.1 .out:Total Excitonic CCSD Energy =  -29.2253373554
6.2 .out:Total Excitonic CCSD Energy =  -29.225341406
6.3 .out:Total Excitonic CCSD Energy =  -29.2253438473
6.4 .out:Total Excitonic CCSD Energy =  -29.2253450752
6.5 .out:Total Excitonic CCSD Energy =  -29.2253454097
6.6 .out:Total Excitonic CCSD Energy =  -29.2253451048
6.7 .out:Total Excitonic CCSD Energy =  -29.2253443593
6.8 .out:Total Excitonic CCSD Energy =  -29.2253433258
6.9 .out:Total Excitonic CCSD Energy =  -29.2253421192
7.0 .out:Total Excitonic CCSD Energy =  -29.2253408241
7.1 .out:Total Excitonic CCSD Energy =  -29.2253395013
7.2 .out:Total Excitonic CCSD Energy =  -29.2253381929
7.3 .out:Total Excitonic CCSD Energy =  -29.2253369272
7.4 .out:Total Excitonic CCSD Energy =  -29.2253357218
7.5 .out:Total Excitonic CCSD Energy =  -29.2253345865
7.6 .out:Total Excitonic CCSD Energy =  -29.2253335261
7.7 .out:Total Excitonic CCSD Energy =  -29.2253325413
7.8 .out:Total Excitonic CCSD Energy =  -29.2253316303
7.9 .out:Total Excitonic CCSD Energy =  -29.22533079
""")
