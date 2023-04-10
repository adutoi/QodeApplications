from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -29.225259498690207
3.1.out:Total Excitonic CCSD Energy =  -29.225103004235272
3.2.out:Total Excitonic CCSD Energy =  -29.225055911787816
3.3.out:Total Excitonic CCSD Energy =  -29.225079462259956
3.4.out:Total Excitonic CCSD Energy =  -29.22514518488783
3.5.out:Total Excitonic CCSD Energy =  -29.225233077708644
3.6.out:Total Excitonic CCSD Energy =  -29.22532947479216
3.7.out:Total Excitonic CCSD Energy =  -29.225425225181674
3.8.out:Total Excitonic CCSD Energy =  -29.22551435500819
3.9.out:Total Excitonic CCSD Energy =  -29.225593158231252
4.0.out:Total Excitonic CCSD Energy =  -29.22565958988072
4.1.out:Total Excitonic CCSD Energy =  -29.225712843812747
4.2.out:Total Excitonic CCSD Energy =  -29.22575303340679
4.3.out:Total Excitonic CCSD Energy =  -29.225780930552645
4.4.out:Total Excitonic CCSD Energy =  -29.225797744548622
4.5.out:Total Excitonic CCSD Energy =  -29.225804936668514
4.6.out:Total Excitonic CCSD Energy =  -29.225804071007346
4.7.out:Total Excitonic CCSD Energy =  -29.22579670176478
4.8.out:Total Excitonic CCSD Energy =  -29.2257842946697
4.9.out:Total Excitonic CCSD Energy =  -29.225768177879157
5.0.out:Total Excitonic CCSD Energy =  -29.225749516329284
5.1.out:Total Excitonic CCSD Energy =  -29.22572930329793
5.2.out:Total Excitonic CCSD Energy =  -29.225708363593196
5.3.out:Total Excitonic CCSD Energy =  -29.2256873638977
5.4.out:Total Excitonic CCSD Energy =  -29.225666827018934
5.5.out:Total Excitonic CCSD Energy =  -29.22564714789128
5.6.out:Total Excitonic CCSD Energy =  -29.225628610012283
5.7.out:Total Excitonic CCSD Energy =  -29.225611401567413
5.8.out:Total Excitonic CCSD Energy =  -29.225595630841116
5.9.out:Total Excitonic CCSD Energy =  -29.22558134068263
6.0.out:Total Excitonic CCSD Energy =  -29.225568521873544
6.1.out:Total Excitonic CCSD Energy =  -29.22555712527345
6.2.out:Total Excitonic CCSD Energy =  -29.22554707264383
6.3.out:Total Excitonic CCSD Energy =  -29.225538266070732
6.4.out:Total Excitonic CCSD Energy =  -29.225530595950246
6.5.out:Total Excitonic CCSD Energy =  -29.22552394754603
6.6.out:Total Excitonic CCSD Energy =  -29.22551820617261
6.7.out:Total Excitonic CCSD Energy =  -29.22551326110758
6.8.out:Total Excitonic CCSD Energy =  -29.22550900836405
6.9.out:Total Excitonic CCSD Energy =  -29.225505352483744
7.0.out:Total Excitonic CCSD Energy =  -29.22550220751478
7.1.out:Total Excitonic CCSD Energy =  -29.22549949734523
7.2.out:Total Excitonic CCSD Energy =  -29.225497155545067
7.3.out:Total Excitonic CCSD Energy =  -29.225495124862068
7.4.out:Total Excitonic CCSD Energy =  -29.225493356489395
7.5.out:Total Excitonic CCSD Energy =  -29.22549180920645
7.6.out:Total Excitonic CCSD Energy =  -29.225490448468584
7.7.out:Total Excitonic CCSD Energy =  -29.22548924550383
7.8.out:Total Excitonic CCSD Energy =  -29.225488176459148
7.9.out:Total Excitonic CCSD Energy =  -29.22548722161612
""")
