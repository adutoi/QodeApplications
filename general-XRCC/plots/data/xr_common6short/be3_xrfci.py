from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-3*atom.energy

data = X( 
parse,
"""\
4.0.out:Total Excitonic FCI  Energy =  -43.83806519245271
4.1.out:Total Excitonic FCI  Energy =  -43.83823428199368
4.2.out:Total Excitonic FCI  Energy =  -43.838377261479344
4.3.out:Total Excitonic FCI  Energy =  -43.83849284861988
4.4.out:Total Excitonic FCI  Energy =  -43.8385817380289
4.5.out:Total Excitonic FCI  Energy =  -43.838645876966034
4.6.out:Total Excitonic FCI  Energy =  -43.83868793950805
4.7.out:Total Excitonic FCI  Energy =  -43.83871095180229
4.8.out:Total Excitonic FCI  Energy =  -43.83871803185533
4.9.out:Total Excitonic FCI  Energy =  -43.83871221451084
""")

#3.0.out:Total Excitonic FCI  Energy =  -43.84069693776314
