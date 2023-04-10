from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0 .out:Total Excitonic CCSD Energy =  -29.222616376258927
3.1 .out:Total Excitonic CCSD Energy =  -29.22310877381688
3.2 .out:Total Excitonic CCSD Energy =  -29.223539525372438
3.3 .out:Total Excitonic CCSD Energy =  -29.223920196628686
3.4 .out:Total Excitonic CCSD Energy =  -29.224256933438227
3.5 .out:Total Excitonic CCSD Energy =  -29.22455321915599
3.6 .out:Total Excitonic CCSD Energy =  -29.224811310448636
3.7 .out:Total Excitonic CCSD Energy =  -29.22503302057342
3.8 .out:Total Excitonic CCSD Energy =  -29.22522017045689
3.9 .out:Total Excitonic CCSD Energy =  -29.2253748395656
4.0 .out:Total Excitonic CCSD Energy =  -29.225499470825923
4.1 .out:Total Excitonic CCSD Energy =  -29.225596863268407
4.2 .out:Total Excitonic CCSD Energy =  -29.225670086033528
4.3 .out:Total Excitonic CCSD Energy =  -29.225722349033735
4.4 .out:Total Excitonic CCSD Energy =  -29.22575686249188
4.5 .out:Total Excitonic CCSD Energy =  -29.22577670983296
4.6 .out:Total Excitonic CCSD Energy =  -29.225784748665692
4.7 .out:Total Excitonic CCSD Energy =  -29.22578354542207
4.8 .out:Total Excitonic CCSD Energy =  -29.225775342285928
4.9 .out:Total Excitonic CCSD Energy =  -29.22576205086472
5.0 .out:Total Excitonic CCSD Energy =  -29.225745265354004
5.1 .out:Total Excitonic CCSD Energy =  -29.225726288052957
5.2 .out:Total Excitonic CCSD Energy =  -29.225706161242762
5.3 .out:Total Excitonic CCSD Energy =  -29.225685701019103
5.4 .out:Total Excitonic CCSD Energy =  -29.22566553022296
5.5 .out:Total Excitonic CCSD Energy =  -29.225646108895642
5.6 .out:Total Excitonic CCSD Energy =  -29.225627761603874
5.7 .out:Total Excitonic CCSD Energy =  -29.225610701555528
5.8 .out:Total Excitonic CCSD Energy =  -29.22559505170844
5.9 .out:Total Excitonic CCSD Energy =  -29.225580863181335
6.0 .out:Total Excitonic CCSD Energy =  -29.22556813126135
6.1 .out:Total Excitonic CCSD Energy =  -29.22555680924821
6.2 .out:Total Excitonic CCSD Energy =  -29.225546820315053
6.3 .out:Total Excitonic CCSD Energy =  -29.225538067528127
6.4 .out:Total Excitonic CCSD Energy =  -29.225530442140247
6.5 .out:Total Excitonic CCSD Energy =  -29.22552383028149
6.6 .out:Total Excitonic CCSD Energy =  -29.22551811817858
6.7 .out:Total Excitonic CCSD Energy =  -29.22551319605085
6.8 .out:Total Excitonic CCSD Energy =  -29.225508960849847
6.9 .out:Total Excitonic CCSD Energy =  -29.225505318015678
7.0 .out:Total Excitonic CCSD Energy =  -29.225502182426023
7.1 .out:Total Excitonic CCSD Energy =  -29.22549947870938
7.2 .out:Total Excitonic CCSD Energy =  -29.225497141077717
7.3 .out:Total Excitonic CCSD Energy =  -29.225495112820305
7.4 .out:Total Excitonic CCSD Energy =  -29.22549334557631
7.5 .out:Total Excitonic CCSD Energy =  -29.225491798483695
7.6 .out:Total Excitonic CCSD Energy =  -29.225490437279866
7.7 .out:Total Excitonic CCSD Energy =  -29.225489233410535
7.8 .out:Total Excitonic CCSD Energy =  -29.225488163185762
7.9 .out:Total Excitonic CCSD Energy =  -29.225487207008015
""")
