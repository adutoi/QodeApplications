from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0 .out:Total Excitonic CCSD Energy =  -29.223987001517543
3.1 .out:Total Excitonic CCSD Energy =  -29.22409726057888
3.2 .out:Total Excitonic CCSD Energy =  -29.224254963870163
3.3 .out:Total Excitonic CCSD Energy =  -29.224437297437095
3.4 .out:Total Excitonic CCSD Energy =  -29.224627855684897
3.5 .out:Total Excitonic CCSD Energy =  -29.224815430361023
3.6 .out:Total Excitonic CCSD Energy =  -29.22499257111608
3.7 .out:Total Excitonic CCSD Energy =  -29.225154433985736
3.8 .out:Total Excitonic CCSD Energy =  -29.22529802037205
3.9 .out:Total Excitonic CCSD Energy =  -29.225421721192635
4.0 .out:Total Excitonic CCSD Energy =  -29.225525038011245
4.1 .out:Total Excitonic CCSD Energy =  -29.225608378897622
4.2 .out:Total Excitonic CCSD Energy =  -29.225672871423875
4.3 .out:Total Excitonic CCSD Energy =  -29.22572017337549
4.4 .out:Total Excitonic CCSD Energy =  -29.225752284963246
4.5 .out:Total Excitonic CCSD Energy =  -29.22577137539499
4.6 .out:Total Excitonic CCSD Energy =  -29.22577963615785
4.7 .out:Total Excitonic CCSD Energy =  -29.225779168257336
4.8 .out:Total Excitonic CCSD Energy =  -29.225771904706313
4.9 .out:Total Excitonic CCSD Energy =  -29.22575956488086
5.0 .out:Total Excitonic CCSD Energy =  -29.22574363467755
5.1 .out:Total Excitonic CCSD Energy =  -29.22572536558264
5.2 .out:Total Excitonic CCSD Energy =  -29.225705786256654
5.3 .out:Total Excitonic CCSD Energy =  -29.225685721426476
5.4 .out:Total Excitonic CCSD Energy =  -29.22566581428802
5.5 .out:Total Excitonic CCSD Energy =  -29.22564654989904
5.6 .out:Total Excitonic CCSD Energy =  -29.22562827806111
5.7 .out:Total Excitonic CCSD Energy =  -29.22561123488509
5.8 .out:Total Excitonic CCSD Energy =  -29.22559556266831
5.9 .out:Total Excitonic CCSD Energy =  -29.225581327937107
6.0 .out:Total Excitonic CCSD Energy =  -29.225568537615466
6.1 .out:Total Excitonic CCSD Energy =  -29.22555715331769
6.2 .out:Total Excitonic CCSD Energy =  -29.225547103784503
6.3 .out:Total Excitonic CCSD Energy =  -29.22553829549157
6.4 .out:Total Excitonic CCSD Energy =  -29.22553062149063
6.5 .out:Total Excitonic CCSD Energy =  -29.225523968569053
6.6 .out:Total Excitonic CCSD Energy =  -29.225518222850404
6.7 .out:Total Excitonic CCSD Energy =  -29.2255132739859
6.8 .out:Total Excitonic CCSD Energy =  -29.225509018110657
6.9 .out:Total Excitonic CCSD Energy =  -29.22550535975321
7.0 .out:Total Excitonic CCSD Energy =  -29.2255022128852
7.1 .out:Total Excitonic CCSD Energy =  -29.225499501295154
7.2 .out:Total Excitonic CCSD Energy =  -29.225497158454008
7.3 .out:Total Excitonic CCSD Energy =  -29.225495127021244
7.4 .out:Total Excitonic CCSD Energy =  -29.225493358116374
7.5 .out:Total Excitonic CCSD Energy =  -29.225491810460486
7.6 .out:Total Excitonic CCSD Energy =  -29.22549044946396
7.7 .out:Total Excitonic CCSD Energy =  -29.225489246321402
7.8 .out:Total Excitonic CCSD Energy =  -29.225488177155228
7.9 .out:Total Excitonic CCSD Energy =  -29.225487222229425
""")
