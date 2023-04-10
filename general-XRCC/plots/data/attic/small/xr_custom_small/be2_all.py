from data.extract import X
from data import atom_small as atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -29.12612384154218
3.1.out:Total Excitonic CCSD Energy =  -29.12831118866673
3.2.out:Total Excitonic CCSD Energy =  -29.1301083504603
3.3.out:Total Excitonic CCSD Energy =  -29.131585100206195
3.4.out:Total Excitonic CCSD Energy =  -29.132798839350148
3.5.out:Total Excitonic CCSD Energy =  -29.133796616306043
3.6.out:Total Excitonic CCSD Energy =  -29.13461689302229
3.7.out:Total Excitonic CCSD Energy =  -29.135291071693864
3.8.out:Total Excitonic CCSD Energy =  -29.13584479321286
3.9.out:Total Excitonic CCSD Energy =  -29.136299023537827
4.0.out:Total Excitonic CCSD Energy =  -29.136670949088597
4.1.out:Total Excitonic CCSD Energy =  -29.136974705389225
4.2.out:Total Excitonic CCSD Energy =  -29.1372219640107
4.3.out:Total Excitonic CCSD Energy =  -29.13742240178533
4.4.out:Total Excitonic CCSD Energy =  -29.1375840739483
4.5.out:Total Excitonic CCSD Energy =  -29.137713709975166
4.6.out:Total Excitonic CCSD Energy =  -29.137816947911627
4.7.out:Total Excitonic CCSD Energy =  -29.137898520234945
4.8.out:Total Excitonic CCSD Energy =  -29.137962401867426
4.9.out:Total Excitonic CCSD Energy =  -29.13801192890418
5.0.out:Total Excitonic CCSD Energy =  -29.13804989488502
5.1.out:Total Excitonic CCSD Energy =  -29.13807862998227
5.2.out:Total Excitonic CCSD Energy =  -29.13810006724783
5.3.out:Total Excitonic CCSD Energy =  -29.13811579903504
5.4.out:Total Excitonic CCSD Energy =  -29.138127125860937
5.5.out:Total Excitonic CCSD Energy =  -29.138135099294466
5.6.out:Total Excitonic CCSD Energy =  -29.138140559931667
5.7.out:Total Excitonic CCSD Energy =  -29.13814417113606
5.8.out:Total Excitonic CCSD Energy =  -29.138146448967593
5.9.out:Total Excitonic CCSD Energy =  -29.13814778856725
6.0.out:Total Excitonic CCSD Energy =  -29.138148487190612
6.1.out:Total Excitonic CCSD Energy =  -29.138148764063054
6.2.out:Total Excitonic CCSD Energy =  -29.138148777243593
6.3.out:Total Excitonic CCSD Energy =  -29.13814863771334
6.4.out:Total Excitonic CCSD Energy =  -29.138148420937704
6.5.out:Total Excitonic CCSD Energy =  -29.13814817617708
6.6.out:Total Excitonic CCSD Energy =  -29.138147933834677
6.7.out:Total Excitonic CCSD Energy =  -29.13814771113337
6.8.out:Total Excitonic CCSD Energy =  -29.13814751640192
6.9.out:Total Excitonic CCSD Energy =  -29.138147352231883
7.0.out:Total Excitonic CCSD Energy =  -29.13814721773848
7.1.out:Total Excitonic CCSD Energy =  -29.13814711012885
7.2.out:Total Excitonic CCSD Energy =  -29.138147025748108
7.3.out:Total Excitonic CCSD Energy =  -29.138146960742894
7.4.out:Total Excitonic CCSD Energy =  -29.138146911453088
7.5.out:Total Excitonic CCSD Energy =  -29.138146874617362
7.6.out:Total Excitonic CCSD Energy =  -29.138146847456355
7.7.out:Total Excitonic CCSD Energy =  -29.13814682768003
7.8.out:Total Excitonic CCSD Energy =  -29.138146813452177
7.9.out:Total Excitonic CCSD Energy =  -29.138146803333363
""")
