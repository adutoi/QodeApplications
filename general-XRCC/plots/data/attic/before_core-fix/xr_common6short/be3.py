from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-3*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -43.83054776975227
3.1.out:Total Excitonic CCSD Energy =  -43.831336040290985
3.2.out:Total Excitonic CCSD Energy =  -43.832234570508355
3.3.out:Total Excitonic CCSD Energy =  -43.83314092955922
3.4.out:Total Excitonic CCSD Energy =  -43.83400094105128
3.5.out:Total Excitonic CCSD Energy =  -43.83478835430829
3.6.out:Total Excitonic CCSD Energy =  -43.83549244686331
3.7.out:Total Excitonic CCSD Energy =  -43.8361108226945
3.8.out:Total Excitonic CCSD Energy =  -43.83664546120151
3.9.out:Total Excitonic CCSD Energy =  -43.83710066435266
4.0.out:Total Excitonic CCSD Energy =  -43.83748202172009
4.1.out:Total Excitonic CCSD Energy =  -43.83779586843016
4.2.out:Total Excitonic CCSD Energy =  -43.8380489553491
4.3.out:Total Excitonic CCSD Energy =  -43.83824820248916
4.4.out:Total Excitonic CCSD Energy =  -43.838400489902185
4.5.out:Total Excitonic CCSD Energy =  -43.83851247856485
4.6.out:Total Excitonic CCSD Energy =  -43.83859046588102
4.7.out:Total Excitonic CCSD Energy =  -43.83864027995644
4.8.out:Total Excitonic CCSD Energy =  -43.838667212201116
4.9.out:Total Excitonic CCSD Energy =  -43.83867598353706
5.0.out:Total Excitonic CCSD Energy =  -43.83867073720831
5.1.out:Total Excitonic CCSD Energy =  -43.83865505094321
5.2.out:Total Excitonic CCSD Energy =  -43.838631962336336
5.3.out:Total Excitonic CCSD Energy =  -43.83860400294871
5.4.out:Total Excitonic CCSD Energy =  -43.838573238167896
5.5.out:Total Excitonic CCSD Energy =  -43.83854131094853
5.6.out:Total Excitonic CCSD Energy =  -43.83850948812352
5.7.out:Total Excitonic CCSD Energy =  -43.83847870814009
5.8.out:Total Excitonic CCSD Energy =  -43.838449629018676
5.9.out:Total Excitonic CCSD Energy =  -43.838422675252346
6.0.out:Total Excitonic CCSD Energy =  -43.838398082358914
6.1.out:Total Excitonic CCSD Energy =  -43.83837593794054
6.2.out:Total Excitonic CCSD Energy =  -43.83835621835751
6.3.out:Total Excitonic CCSD Energy =  -43.83833882045658
6.4.out:Total Excitonic CCSD Energy =  -43.83832358813896
6.5.out:Total Excitonic CCSD Energy =  -43.83831033386296
6.6.out:Total Excitonic CCSD Energy =  -43.83829885542161
6.7.out:Total Excitonic CCSD Energy =  -43.83828894850319
6.8.out:Total Excitonic CCSD Energy =  -43.83828041562897
6.9.out:Total Excitonic CCSD Energy =  -43.838273072096875
7.0.out:Total Excitonic CCSD Energy =  -43.83826674953046
7.1.out:Total Excitonic CCSD Energy =  -43.83826129759147
7.2.out:Total Excitonic CCSD Energy =  -43.838256584335085
7.3.out:Total Excitonic CCSD Energy =  -43.83825249561731
7.4.out:Total Excitonic CCSD Energy =  -43.83824893388333
7.5.out:Total Excitonic CCSD Energy =  -43.838245816596256
7.6.out:Total Excitonic CCSD Energy =  -43.838243074502174
7.7.out:Total Excitonic CCSD Energy =  -43.8382406498699
7.8.out:Total Excitonic CCSD Energy =  -43.8382384948051
7.9.out:Total Excitonic CCSD Energy =  -43.83823656969537
""")
