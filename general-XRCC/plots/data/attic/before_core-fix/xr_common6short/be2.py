from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -29.223987001519475
3.1.out:Total Excitonic CCSD Energy =  -29.22409726057883
3.2.out:Total Excitonic CCSD Energy =  -29.224254963869953
3.3.out:Total Excitonic CCSD Energy =  -29.22443729743706
3.4.out:Total Excitonic CCSD Energy =  -29.22462785568512
3.5.out:Total Excitonic CCSD Energy =  -29.2248154303613
3.6.out:Total Excitonic CCSD Energy =  -29.22499257111624
3.7.out:Total Excitonic CCSD Energy =  -29.225154433985896
3.8.out:Total Excitonic CCSD Energy =  -29.22529802037211
3.9.out:Total Excitonic CCSD Energy =  -29.225421721192628
4.0.out:Total Excitonic CCSD Energy =  -29.22552503801123
4.1.out:Total Excitonic CCSD Energy =  -29.2256083788976
4.2.out:Total Excitonic CCSD Energy =  -29.22567287142385
4.3.out:Total Excitonic CCSD Energy =  -29.225720173375507
4.4.out:Total Excitonic CCSD Energy =  -29.22575228496329
4.5.out:Total Excitonic CCSD Energy =  -29.22577137539499
4.6.out:Total Excitonic CCSD Energy =  -29.22577963615789
4.7.out:Total Excitonic CCSD Energy =  -29.22577916825735
4.8.out:Total Excitonic CCSD Energy =  -29.22577190470626
4.9.out:Total Excitonic CCSD Energy =  -29.225759564880818
5.0.out:Total Excitonic CCSD Energy =  -29.22574363467756
5.1.out:Total Excitonic CCSD Energy =  -29.22572536558265
5.2.out:Total Excitonic CCSD Energy =  -29.225705786256665
5.3.out:Total Excitonic CCSD Energy =  -29.22568572142648
5.4.out:Total Excitonic CCSD Energy =  -29.225665814288035
5.5.out:Total Excitonic CCSD Energy =  -29.22564654989908
5.6.out:Total Excitonic CCSD Energy =  -29.225628278061116
5.7.out:Total Excitonic CCSD Energy =  -29.225611234885076
5.8.out:Total Excitonic CCSD Energy =  -29.225595562668335
5.9.out:Total Excitonic CCSD Energy =  -29.22558132793709
6.0.out:Total Excitonic CCSD Energy =  -29.22556853761545
6.1.out:Total Excitonic CCSD Energy =  -29.225557153317688
6.2.out:Total Excitonic CCSD Energy =  -29.225547103784482
6.3.out:Total Excitonic CCSD Energy =  -29.225538295491592
6.4.out:Total Excitonic CCSD Energy =  -29.225530621490627
6.5.out:Total Excitonic CCSD Energy =  -29.225523968569057
6.6.out:Total Excitonic CCSD Energy =  -29.225518222850386
6.7.out:Total Excitonic CCSD Energy =  -29.225513273985865
6.8.out:Total Excitonic CCSD Energy =  -29.225509018110678
6.9.out:Total Excitonic CCSD Energy =  -29.22550535975321
7.0.out:Total Excitonic CCSD Energy =  -29.225502212885168
7.1.out:Total Excitonic CCSD Energy =  -29.22549950129517
7.2.out:Total Excitonic CCSD Energy =  -29.225497158454015
7.3.out:Total Excitonic CCSD Energy =  -29.22549512702126
7.4.out:Total Excitonic CCSD Energy =  -29.22549335811639
7.5.out:Total Excitonic CCSD Energy =  -29.22549181046049
7.6.out:Total Excitonic CCSD Energy =  -29.225490449463972
7.7.out:Total Excitonic CCSD Energy =  -29.225489246321402
7.8.out:Total Excitonic CCSD Energy =  -29.22548817715523
7.9.out:Total Excitonic CCSD Energy =  -29.225487222229457
""")
