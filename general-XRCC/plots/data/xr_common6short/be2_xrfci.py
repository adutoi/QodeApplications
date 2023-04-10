from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic FCI  Energy =  -29.22525949869047
3.1.out:Total Excitonic FCI  Energy =  -29.22510300423545
3.2.out:Total Excitonic FCI  Energy =  -29.225055911788004
3.3.out:Total Excitonic FCI  Energy =  -29.225079462259895
3.4.out:Total Excitonic FCI  Energy =  -29.225145184887733
3.5.out:Total Excitonic FCI  Energy =  -29.22523307770852
3.6.out:Total Excitonic FCI  Energy =  -29.225329474792225
3.7.out:Total Excitonic FCI  Energy =  -29.225425225181716
3.8.out:Total Excitonic FCI  Energy =  -29.225514355008244
3.9.out:Total Excitonic FCI  Energy =  -29.225593158231295
4.0.out:Total Excitonic FCI  Energy =  -29.225659589880767
4.1.out:Total Excitonic FCI  Energy =  -29.225712843812865
4.2.out:Total Excitonic FCI  Energy =  -29.225753033406793
4.3.out:Total Excitonic FCI  Energy =  -29.225780930552673
4.4.out:Total Excitonic FCI  Energy =  -29.225797744548643
4.5.out:Total Excitonic FCI  Energy =  -29.225804936668556
4.6.out:Total Excitonic FCI  Energy =  -29.225804071007357
4.7.out:Total Excitonic FCI  Energy =  -29.22579670176488
4.8.out:Total Excitonic FCI  Energy =  -29.225784294669758
4.9.out:Total Excitonic FCI  Energy =  -29.225768177879218
5.0.out:Total Excitonic FCI  Energy =  -29.22574951632932
5.1.out:Total Excitonic FCI  Energy =  -29.225729303298
5.2.out:Total Excitonic FCI  Energy =  -29.225708363593323
5.3.out:Total Excitonic FCI  Energy =  -29.22568736389774
5.4.out:Total Excitonic FCI  Energy =  -29.22566682701901
5.5.out:Total Excitonic FCI  Energy =  -29.22564714789131
5.6.out:Total Excitonic FCI  Energy =  -29.22562861001226
5.7.out:Total Excitonic FCI  Energy =  -29.225611401567463
5.8.out:Total Excitonic FCI  Energy =  -29.225595630841173
5.9.out:Total Excitonic FCI  Energy =  -29.225581340682687
6.0.out:Total Excitonic FCI  Energy =  -29.225568521873623
6.1.out:Total Excitonic FCI  Energy =  -29.225557125273454
6.2.out:Total Excitonic FCI  Energy =  -29.225547072643877
6.3.out:Total Excitonic FCI  Energy =  -29.225538266070757
6.4.out:Total Excitonic FCI  Energy =  -29.225530595950257
6.5.out:Total Excitonic FCI  Energy =  -29.225523947546012
6.6.out:Total Excitonic FCI  Energy =  -29.22551820617262
6.7.out:Total Excitonic FCI  Energy =  -29.22551326110759
6.8.out:Total Excitonic FCI  Energy =  -29.225509008364078
6.9.out:Total Excitonic FCI  Energy =  -29.225505352483793
7.0.out:Total Excitonic FCI  Energy =  -29.225502207514868
7.1.out:Total Excitonic FCI  Energy =  -29.22549949734529
7.2.out:Total Excitonic FCI  Energy =  -29.225497155545167
7.3.out:Total Excitonic FCI  Energy =  -29.2254951248621
7.4.out:Total Excitonic FCI  Energy =  -29.22549335648943
7.5.out:Total Excitonic FCI  Energy =  -29.225491809206453
7.6.out:Total Excitonic FCI  Energy =  -29.22549044846866
7.7.out:Total Excitonic FCI  Energy =  -29.22548924550383
7.8.out:Total Excitonic FCI  Energy =  -29.22548817645919
7.9.out:Total Excitonic FCI  Energy =  -29.225487221616092
""")
