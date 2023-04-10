from data.extract import X
from data import atom_small as atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-2*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -29.126379265596928
3.1.out:Total Excitonic CCSD Energy =  -29.12854080605559
3.2.out:Total Excitonic CCSD Energy =  -29.1303076369318
3.3.out:Total Excitonic CCSD Energy =  -29.13175253280571
3.4.out:Total Excitonic CCSD Energy =  -29.13293534387237
3.5.out:Total Excitonic CCSD Energy =  -29.133904849569987
3.6.out:Total Excitonic CCSD Energy =  -29.13470052201499
3.7.out:Total Excitonic CCSD Energy =  -29.1353541572501
3.8.out:Total Excitonic CCSD Energy =  -29.13589132997792
3.9.out:Total Excitonic CCSD Energy =  -29.136332643313217
4.0.out:Total Excitonic CCSD Energy =  -29.136694766178998
4.1.out:Total Excitonic CCSD Energy =  -29.136991269406906
4.2.out:Total Excitonic CCSD Energy =  -29.137233283858855
4.3.out:Total Excitonic CCSD Energy =  -29.13743000958162
4.4.out:Total Excitonic CCSD Energy =  -29.137589105493817
4.5.out:Total Excitonic CCSD Energy =  -29.137716986327906
4.6.out:Total Excitonic CCSD Energy =  -29.137819049257487
4.7.out:Total Excitonic CCSD Energy =  -29.13789984808407
4.8.out:Total Excitonic CCSD Energy =  -29.13796322873105
4.9.out:Total Excitonic CCSD Energy =  -29.13801243637663
5.0.out:Total Excitonic CCSD Energy =  -29.138050201871984
5.1.out:Total Excitonic CCSD Energy =  -29.138078813033644
5.2.out:Total Excitonic CCSD Energy =  -29.138100174838357
5.3.out:Total Excitonic CCSD Energy =  -29.138115861367538
5.4.out:Total Excitonic CCSD Energy =  -29.138127161455003
5.5.out:Total Excitonic CCSD Energy =  -29.13813511932719
5.6.out:Total Excitonic CCSD Energy =  -29.13814057104317
5.7.out:Total Excitonic CCSD Energy =  -29.138144177209604
5.8.out:Total Excitonic CCSD Energy =  -29.138146452238814
5.9.out:Total Excitonic CCSD Energy =  -29.138147790303183
6.0.out:Total Excitonic CCSD Energy =  -29.13814848809814
6.1.out:Total Excitonic CCSD Energy =  -29.138148764530396
6.2.out:Total Excitonic CCSD Energy =  -29.13814877748062
6.3.out:Total Excitonic CCSD Energy =  -29.138148637831716
6.4.out:Total Excitonic CCSD Energy =  -29.138148420995908
6.5.out:Total Excitonic CCSD Energy =  -29.138148176205245
6.6.out:Total Excitonic CCSD Energy =  -29.13814793384809
6.7.out:Total Excitonic CCSD Energy =  -29.138147711139652
6.8.out:Total Excitonic CCSD Energy =  -29.138147516404814
6.9.out:Total Excitonic CCSD Energy =  -29.138147352233194
7.0.out:Total Excitonic CCSD Energy =  -29.138147217739064
7.1.out:Total Excitonic CCSD Energy =  -29.138147110129108
7.2.out:Total Excitonic CCSD Energy =  -29.138147025748214
7.3.out:Total Excitonic CCSD Energy =  -29.13814696074294
7.4.out:Total Excitonic CCSD Energy =  -29.13814691145311
7.5.out:Total Excitonic CCSD Energy =  -29.138146874617366
7.6.out:Total Excitonic CCSD Energy =  -29.138146847456355
7.7.out:Total Excitonic CCSD Energy =  -29.13814682768002
7.8.out:Total Excitonic CCSD Energy =  -29.138146813452177
7.9.out:Total Excitonic CCSD Energy =  -29.138146803333356
""")
