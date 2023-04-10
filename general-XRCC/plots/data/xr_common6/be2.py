from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -29.223888412133654
3.1.out:Total Excitonic CCSD Energy =  -29.224113250827216
3.2.out:Total Excitonic CCSD Energy =  -29.224339329918365
3.3.out:Total Excitonic CCSD Energy =  -29.22456169641507
3.4.out:Total Excitonic CCSD Energy =  -29.2247741133955
3.5.out:Total Excitonic CCSD Energy =  -29.224971122885528
3.6.out:Total Excitonic CCSD Energy =  -29.225148722750138
3.7.out:Total Excitonic CCSD Energy =  -29.225304432177378
3.8.out:Total Excitonic CCSD Energy =  -29.225437133035314
3.9.out:Total Excitonic CCSD Energy =  -29.22554684750446
4.0.out:Total Excitonic CCSD Energy =  -29.225634505191582
4.1.out:Total Excitonic CCSD Energy =  -29.225701714374754
4.2.out:Total Excitonic CCSD Energy =  -29.225750544151094
4.3.out:Total Excitonic CCSD Energy =  -29.225783325448152
4.4.out:Total Excitonic CCSD Energy =  -29.2258024796331
4.5.out:Total Excitonic CCSD Energy =  -29.225810381440663
4.6.out:Total Excitonic CCSD Energy =  -29.22580925899977
4.7.out:Total Excitonic CCSD Energy =  -29.22580112945496
4.8.out:Total Excitonic CCSD Energy =  -29.22578776534744
4.9.out:Total Excitonic CCSD Energy =  -29.22577068506194
5.0.out:Total Excitonic CCSD Energy =  -29.22575116024589
5.1.out:Total Excitonic CCSD Energy =  -29.225730233790316
5.2.out:Total Excitonic CCSD Energy =  -29.2257087432509
5.3.out:Total Excitonic CCSD Energy =  -29.22568734605961
""")
