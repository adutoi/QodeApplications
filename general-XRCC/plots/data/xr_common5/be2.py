from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -29.225058728706067
3.1.out:Total Excitonic CCSD Energy =  -29.22493043910368
3.2.out:Total Excitonic CCSD Energy =  -29.22491339241474
3.3.out:Total Excitonic CCSD Energy =  -29.224964977456583
3.4.out:Total Excitonic CCSD Energy =  -29.225055264518932
3.5.out:Total Excitonic CCSD Energy =  -29.225163909398972
3.6.out:Total Excitonic CCSD Energy =  -29.225277372333032
3.7.out:Total Excitonic CCSD Energy =  -29.22538680726633
3.8.out:Total Excitonic CCSD Energy =  -29.225486612336415
3.9.out:Total Excitonic CCSD Energy =  -29.2255734823896
4.0.out:Total Excitonic CCSD Energy =  -29.22564578568183
4.1.out:Total Excitonic CCSD Energy =  -29.225703126844433
4.2.out:Total Excitonic CCSD Energy =  -29.22574601076147
4.3.out:Total Excitonic CCSD Energy =  -29.225775565162216
4.4.out:Total Excitonic CCSD Energy =  -29.225793306858375
4.5.out:Total Excitonic CCSD Energy =  -29.22580094931024
4.6.out:Total Excitonic CCSD Energy =  -29.225800252346463
4.7.out:Total Excitonic CCSD Energy =  -29.225792913150833
4.8.out:Total Excitonic CCSD Energy =  -29.225780494618913
4.9.out:Total Excitonic CCSD Energy =  -29.225764384789336
5.0.out:Total Excitonic CCSD Energy =  -29.225745779991936
5.1.out:Total Excitonic CCSD Energy =  -29.22572568455257
5.2.out:Total Excitonic CCSD Energy =  -29.225704920940462
5.3.out:Total Excitonic CCSD Energy =  -29.225684145671426
5.4.out:Total Excitonic CCSD Energy =  -29.22566386772656
5.5.out:Total Excitonic CCSD Energy =  -29.225644467467827
5.6.out:Total Excitonic CCSD Energy =  -29.225626214940636
5.7.out:Total Excitonic CCSD Energy =  -29.22560928704224
5.8.out:Total Excitonic CCSD Energy =  -29.225593783369263
5.9.out:Total Excitonic CCSD Energy =  -29.22557974070177
6.0.out:Total Excitonic CCSD Energy =  -29.225567146130174
6.1.out:Total Excitonic CCSD Energy =  -29.22555594882041
6.2.out:Total Excitonic CCSD Energy =  -29.225546070403965
6.3.out:Total Excitonic CCSD Energy =  -29.225537413976966
6.4.out:Total Excitonic CCSD Energy =  -29.225529871705575
6.5.out:Total Excitonic CCSD Energy =  -29.22552333106563
6.6.out:Total Excitonic CCSD Energy =  -29.225517679776953
6.7.out:Total Excitonic CCSD Energy =  -29.2255128095313
6.8.out:Total Excitonic CCSD Energy =  -29.225508618639246
6.9.out:Total Excitonic CCSD Energy =  -29.22550501374418
7.0.out:Total Excitonic CCSD Energy =  -29.225501910759917
7.1.out:Total Excitonic CCSD Energy =  -29.22549923518957
7.2.out:Total Excitonic CCSD Energy =  -29.225496921973622
7.3.out:Total Excitonic CCSD Energy =  -29.225494915002983
7.4.out:Total Excitonic CCSD Energy =  -29.225493166410814
7.5.out:Total Excitonic CCSD Energy =  -29.225491635740024
7.6.out:Total Excitonic CCSD Energy =  -29.225490289059817
7.7.out:Total Excitonic CCSD Energy =  -29.225489098088755
7.8.out:Total Excitonic CCSD Energy =  -29.225488039362077
7.9.out:Total Excitonic CCSD Energy =  -29.225487093468928
""")
