from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-3*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -43.834074897597006
3.1.out:Total Excitonic CCSD Energy =  -43.833635311738504
3.2.out:Total Excitonic CCSD Energy =  -43.83373725237254
3.3.out:Total Excitonic CCSD Energy =  -43.83414831896927
3.4.out:Total Excitonic CCSD Energy =  -43.83470973073992
3.5.out:Total Excitonic CCSD Energy =  -43.835319286073656
3.6.out:Total Excitonic CCSD Energy =  -43.83591528128525
3.7.out:Total Excitonic CCSD Energy =  -43.836463422025886
3.8.out:Total Excitonic CCSD Energy =  -43.83694714775638
3.9.out:Total Excitonic CCSD Energy =  -43.837360916076875
4.0.out:Total Excitonic CCSD Energy =  -43.83770572855457
4.1.out:Total Excitonic CCSD Energy =  -43.83798623130738
4.2.out:Total Excitonic CCSD Energy =  -43.838208878256566
4.3.out:Total Excitonic CCSD Energy =  -43.83838079749199
""")
