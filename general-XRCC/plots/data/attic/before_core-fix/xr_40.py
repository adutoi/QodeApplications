from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0 .out:Total Excitonic CCSD Energy =  -29.2261118213
3.1 .out:Total Excitonic CCSD Energy =  -29.2257897444
3.2 .out:Total Excitonic CCSD Energy =  -29.2255705002
3.3 .out:Total Excitonic CCSD Energy =  -29.2254304196
3.4 .out:Total Excitonic CCSD Energy =  -29.2253512648
3.5 .out:Total Excitonic CCSD Energy =  -29.2253183702
3.6 .out:Total Excitonic CCSD Energy =  -29.2253194614
3.7 .out:Total Excitonic CCSD Energy =  -29.2253440585
3.8 .out:Total Excitonic CCSD Energy =  -29.2253832742
3.9 .out:Total Excitonic CCSD Energy =  -29.2254297999
4.0 .out:Total Excitonic CCSD Energy =  -29.2254779217
4.1 .out:Total Excitonic CCSD Energy =  -29.2255234751
4.2 .out:Total Excitonic CCSD Energy =  -29.2255637113
4.3 .out:Total Excitonic CCSD Energy =  -29.2255970866
4.4 .out:Total Excitonic CCSD Energy =  -29.2256230105
4.5 .out:Total Excitonic CCSD Energy =  -29.2256415876
4.6 .out:Total Excitonic CCSD Energy =  -29.2256533825
4.7 .out:Total Excitonic CCSD Energy =  -29.2256592249
4.8 .out:Total Excitonic CCSD Energy =  -29.2256600618
4.9 .out:Total Excitonic CCSD Energy =  -29.2256568527
5.0 .out:Total Excitonic CCSD Energy =  -29.2256505044
5.1 .out:Total Excitonic CCSD Energy =  -29.2256418333
5.2 .out:Total Excitonic CCSD Energy =  -29.2256315495
5.3 .out:Total Excitonic CCSD Energy =  -29.2256202527
5.4 .out:Total Excitonic CCSD Energy =  -29.2256084379
5.5 .out:Total Excitonic CCSD Energy =  -29.225596503
5.6 .out:Total Excitonic CCSD Energy =  -29.2255847599
5.7 .out:Total Excitonic CCSD Energy =  -29.2255734445
5.8 .out:Total Excitonic CCSD Energy =  -29.2255627277
5.9 .out:Total Excitonic CCSD Energy =  -29.2255527244
6.0 .out:Total Excitonic CCSD Energy =  -29.2255435032
6.1 .out:Total Excitonic CCSD Energy =  -29.2255350947
6.2 .out:Total Excitonic CCSD Energy =  -29.2255274991
6.3 .out:Total Excitonic CCSD Energy =  -29.2255206934
6.4 .out:Total Excitonic CCSD Energy =  -29.2255146376
6.5 .out:Total Excitonic CCSD Energy =  -29.22550928
6.6 .out:Total Excitonic CCSD Energy =  -29.2255045621
6.7 .out:Total Excitonic CCSD Energy =  -29.225500422
6.8 .out:Total Excitonic CCSD Energy =  -29.2254967978
6.9 .out:Total Excitonic CCSD Energy =  -29.2254936294
7.0 .out:Total Excitonic CCSD Energy =  -29.2254908602
7.1 .out:Total Excitonic CCSD Energy =  -29.2254884382
7.2 .out:Total Excitonic CCSD Energy =  -29.2254863166
7.3 .out:Total Excitonic CCSD Energy =  -29.2254844537
7.4 .out:Total Excitonic CCSD Energy =  -29.225482813
7.5 .out:Total Excitonic CCSD Energy =  -29.225481363
7.6 .out:Total Excitonic CCSD Energy =  -29.2254800766
7.7 .out:Total Excitonic CCSD Energy =  -29.2254789307
7.8 .out:Total Excitonic CCSD Energy =  -29.2254779059
7.9 .out:Total Excitonic CCSD Energy =  -29.2254769857
""")
