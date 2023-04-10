from data.extract import X
from data import atom_small as atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-3*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -43.68287898230639
3.1.out:Total Excitonic CCSD Energy =  -43.687364872625736
3.2.out:Total Excitonic CCSD Energy =  -43.69105041021732
3.3.out:Total Excitonic CCSD Energy =  -43.69407189494409
3.4.out:Total Excitonic CCSD Energy =  -43.69654530544444
3.5.out:Total Excitonic CCSD Energy =  -43.6985683041853
3.6.out:Total Excitonic CCSD Energy =  -43.70022221651365
3.7.out:Total Excitonic CCSD Energy =  -43.70157409028445
3.8.out:Total Excitonic CCSD Energy =  -43.70267878120294
3.9.out:Total Excitonic CCSD Energy =  -43.703580957647006
4.0.out:Total Excitonic CCSD Energy =  -43.70431692900289
4.1.out:Total Excitonic CCSD Energy =  -43.70491623750999
4.2.out:Total Excitonic CCSD Energy =  -43.70540299250036
4.3.out:Total Excitonic CCSD Energy =  -43.70579695627019
4.4.out:Total Excitonic CCSD Energy =  -43.70611440927326
4.5.out:Total Excitonic CCSD Energy =  -43.70636883016308
4.6.out:Total Excitonic CCSD Energy =  -43.70657142662122
4.7.out:Total Excitonic CCSD Energy =  -43.70673154918244
4.8.out:Total Excitonic CCSD Energy =  -43.706857014928396
4.9.out:Total Excitonic CCSD Energy =  -43.70695436251852
5.0.out:Total Excitonic CCSD Energy =  -43.70702905529369
5.1.out:Total Excitonic CCSD Energy =  -43.70708564531631
5.2.out:Total Excitonic CCSD Energy =  -43.70712790813138
5.3.out:Total Excitonic CCSD Energy =  -43.707158955586685
5.4.out:Total Excitonic CCSD Energy =  -43.70718133210007
5.5.out:Total Excitonic CCSD Energy =  -43.70719709820327
5.6.out:Total Excitonic CCSD Energy =  -43.70720790397627
5.7.out:Total Excitonic CCSD Energy =  -43.70721505406748
5.8.out:Total Excitonic CCSD Energy =  -43.70721956534988
5.9.out:Total Excitonic CCSD Energy =  -43.7072222178482
6.0.out:Total Excitonic CCSD Energy =  -43.70722359934449
6.1.out:Total Excitonic CCSD Energy =  -43.70722414398062
6.2.out:Total Excitonic CCSD Energy =  -43.70722416517934
6.3.out:Total Excitonic CCSD Energy =  -43.707223883254706
6.4.out:Total Excitonic CCSD Energy =  -43.70722344815165
6.5.out:Total Excitonic CCSD Energy =  -43.7072229578134
6.6.out:Total Excitonic CCSD Energy =  -43.70722247271518
6.7.out:Total Excitonic CCSD Energy =  -43.70722202711653
6.8.out:Total Excitonic CCSD Energy =  -43.7072216375724
6.9.out:Total Excitonic CCSD Energy =  -43.707221309210375
7.0.out:Total Excitonic CCSD Energy =  -43.70722104023142
7.1.out:Total Excitonic CCSD Energy =  -43.70722082503454
7.2.out:Total Excitonic CCSD Energy =  -43.707220656302134
7.3.out:Total Excitonic CCSD Energy =  -43.7072205263235
7.4.out:Total Excitonic CCSD Energy =  -43.70722042777659
7.5.out:Total Excitonic CCSD Energy =  -43.70722035413777
7.6.out:Total Excitonic CCSD Energy =  -43.70722029984787
7.7.out:Total Excitonic CCSD Energy =  -43.707220260326714
7.8.out:Total Excitonic CCSD Energy =  -43.70722023190173
7.9.out:Total Excitonic CCSD Energy =  -43.707220211694136
""")
