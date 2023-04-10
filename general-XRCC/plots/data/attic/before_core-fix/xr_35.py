from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0 .out:Total Excitonic CCSD Energy =  -29.2259946749
3.1 .out:Total Excitonic CCSD Energy =  -29.2253997764
3.2 .out:Total Excitonic CCSD Energy =  -29.2249656471
3.3 .out:Total Excitonic CCSD Energy =  -29.2246746137
3.4 .out:Total Excitonic CCSD Energy =  -29.224504624
3.5 .out:Total Excitonic CCSD Energy =  -29.2244319466
3.6 .out:Total Excitonic CCSD Energy =  -29.2244332808
3.7 .out:Total Excitonic CCSD Energy =  -29.2244872654
3.8 .out:Total Excitonic CCSD Energy =  -29.2245754339
3.9 .out:Total Excitonic CCSD Energy =  -29.2246826836
4.0 .out:Total Excitonic CCSD Energy =  -29.2247973438
4.1 .out:Total Excitonic CCSD Energy =  -29.2249109445
4.2 .out:Total Excitonic CCSD Energy =  -29.2250177919
4.3 .out:Total Excitonic CCSD Energy =  -29.2251144512
4.4 .out:Total Excitonic CCSD Energy =  -29.2251992156
4.5 .out:Total Excitonic CCSD Energy =  -29.2252716202
4.6 .out:Total Excitonic CCSD Energy =  -29.2253320308
4.7 .out:Total Excitonic CCSD Energy =  -29.2253813214
4.8 .out:Total Excitonic CCSD Energy =  -29.2254206355
4.9 .out:Total Excitonic CCSD Energy =  -29.2254512214
5.0 .out:Total Excitonic CCSD Energy =  -29.2254743258
5.1 .out:Total Excitonic CCSD Energy =  -29.2254911303
5.2 .out:Total Excitonic CCSD Energy =  -29.2255027175
5.3 .out:Total Excitonic CCSD Energy =  -29.2255100567
5.4 .out:Total Excitonic CCSD Energy =  -29.2255140008
5.5 .out:Total Excitonic CCSD Energy =  -29.2255152896
5.6 .out:Total Excitonic CCSD Energy =  -29.2255145565
5.7 .out:Total Excitonic CCSD Energy =  -29.2255123359
5.8 .out:Total Excitonic CCSD Energy =  -29.2255090723
5.9 .out:Total Excitonic CCSD Energy =  -29.2255051285
6.0 .out:Total Excitonic CCSD Energy =  -29.2255007947
6.1 .out:Total Excitonic CCSD Energy =  -29.2254962973
6.2 .out:Total Excitonic CCSD Energy =  -29.2254918072
6.3 .out:Total Excitonic CCSD Energy =  -29.2254874487
6.4 .out:Total Excitonic CCSD Energy =  -29.2254833069
6.5 .out:Total Excitonic CCSD Energy =  -29.2254794356
6.6 .out:Total Excitonic CCSD Energy =  -29.2254758639
6.7 .out:Total Excitonic CCSD Energy =  -29.2254726021
6.8 .out:Total Excitonic CCSD Energy =  -29.2254696468
6.9 .out:Total Excitonic CCSD Energy =  -29.2254669849
7.0 .out:Total Excitonic CCSD Energy =  -29.2254645976
7.1 .out:Total Excitonic CCSD Energy =  -29.2254624624
7.2 .out:Total Excitonic CCSD Energy =  -29.2254605556
7.3 .out:Total Excitonic CCSD Energy =  -29.2254588533
7.4 .out:Total Excitonic CCSD Energy =  -29.2254573328
7.5 .out:Total Excitonic CCSD Energy =  -29.225455973
7.6 .out:Total Excitonic CCSD Energy =  -29.2254547546
7.7 .out:Total Excitonic CCSD Energy =  -29.2254536604
7.8 .out:Total Excitonic CCSD Energy =  -29.2254526752
7.9 .out:Total Excitonic CCSD Energy =  -29.2254517859
""")
