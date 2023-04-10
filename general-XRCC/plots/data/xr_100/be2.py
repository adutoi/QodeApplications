from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
4.0.out:Total Excitonic CCSD Energy =  -29.22562642300867
4.1.out:Total Excitonic CCSD Energy =  -29.225698913086145
4.2.out:Total Excitonic CCSD Energy =  -29.2257513815759
""")
