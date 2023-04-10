from data.extract import X
from data import atom_small as atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-3*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -43.68354053531014
3.1.out:Total Excitonic CCSD Energy =  -43.687936640757485
3.2.out:Total Excitonic CCSD Energy =  -43.69153104188041
3.3.out:Total Excitonic CCSD Energy =  -43.69446517701966
3.4.out:Total Excitonic CCSD Energy =  -43.69685891427122
3.5.out:Total Excitonic CCSD Energy =  -43.698812324276105
3.6.out:Total Excitonic CCSD Energy =  -43.7004077409958
3.7.out:Total Excitonic CCSD Energy =  -43.70171209537459
3.8.out:Total Excitonic CCSD Energy =  -43.702779349649205
3.9.out:Total Excitonic CCSD Energy =  -43.70365283905467
4.0.out:Total Excitonic CCSD Energy =  -43.70436737500521
4.1.out:Total Excitonic CCSD Energy =  -43.70495103159318
4.2.out:Total Excitonic CCSD Energy =  -43.70542659775352
4.3.out:Total Excitonic CCSD Energy =  -43.70581271914107
4.4.out:Total Excitonic CCSD Energy =  -43.70612477568396
4.5.out:Total Excitonic CCSD Energy =  -43.70637554721491
4.6.out:Total Excitonic CCSD Energy =  -43.706575716346094
4.7.out:Total Excitonic CCSD Energy =  -43.706734249918085
4.8.out:Total Excitonic CCSD Energy =  -43.70685869142396
4.9.out:Total Excitonic CCSD Energy =  -43.70695538870621
5.0.out:Total Excitonic CCSD Energy =  -43.70702967468926
5.1.out:Total Excitonic CCSD Energy =  -43.70708601397202
5.2.out:Total Excitonic CCSD Energy =  -43.70712812448616
5.3.out:Total Excitonic CCSD Energy =  -43.70715908077851
5.4.out:Total Excitonic CCSD Energy =  -43.70718140351905
5.5.out:Total Excitonic CCSD Energy =  -43.707197138367455
5.6.out:Total Excitonic CCSD Energy =  -43.70720792624051
5.7.out:Total Excitonic CCSD Energy =  -43.70721506623138
5.8.out:Total Excitonic CCSD Energy =  -43.70721957189902
5.9.out:Total Excitonic CCSD Energy =  -43.70722222132266
6.0.out:Total Excitonic CCSD Energy =  -43.70722360116053
6.1.out:Total Excitonic CCSD Energy =  -43.70722414491566
6.2.out:Total Excitonic CCSD Energy =  -43.70722416565352
6.3.out:Total Excitonic CCSD Energy =  -43.7072238834915
6.4.out:Total Excitonic CCSD Energy =  -43.70722344826808
6.5.out:Total Excitonic CCSD Energy =  -43.70722295786974
6.6.out:Total Excitonic CCSD Energy =  -43.70722247274201
6.7.out:Total Excitonic CCSD Energy =  -43.707222027129106
6.8.out:Total Excitonic CCSD Energy =  -43.707221637578186
6.9.out:Total Excitonic CCSD Energy =  -43.70722130921299
7.0.out:Total Excitonic CCSD Energy =  -43.70722104023258
7.1.out:Total Excitonic CCSD Energy =  -43.70722082503505
7.2.out:Total Excitonic CCSD Energy =  -43.70722065630235
7.3.out:Total Excitonic CCSD Energy =  -43.707220526323596
7.4.out:Total Excitonic CCSD Energy =  -43.707220427776626
7.5.out:Total Excitonic CCSD Energy =  -43.70722035413778
7.6.out:Total Excitonic CCSD Energy =  -43.70722029984787
7.7.out:Total Excitonic CCSD Energy =  -43.70722026032672
7.8.out:Total Excitonic CCSD Energy =  -43.70722023190173
7.9.out:Total Excitonic CCSD Energy =  -43.707220211694136
""")
