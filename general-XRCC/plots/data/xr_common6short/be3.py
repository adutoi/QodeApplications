from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-3*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic CCSD Energy =  -43.83452289684995
3.1.out:Total Excitonic CCSD Energy =  -43.83431403020066
3.2.out:Total Excitonic CCSD Energy =  -43.83446190495307
3.3.out:Total Excitonic CCSD Energy =  -43.83481547024722
3.4.out:Total Excitonic CCSD Energy =  -43.83527100686974
3.5.out:Total Excitonic CCSD Energy =  -43.83576096006832
3.6.out:Total Excitonic CCSD Energy =  -43.83624347173384
3.7.out:Total Excitonic CCSD Energy =  -43.83669402550965
3.8.out:Total Excitonic CCSD Energy =  -43.83709942139181
3.9.out:Total Excitonic CCSD Energy =  -43.83745370647101
4.0.out:Total Excitonic CCSD Energy =  -43.83775552604303
4.1.out:Total Excitonic CCSD Energy =  -43.83800641679042
4.2.out:Total Excitonic CCSD Energy =  -43.83820969296372
4.3.out:Total Excitonic CCSD Energy =  -43.83836970024353
4.4.out:Total Excitonic CCSD Energy =  -43.83849130303085
4.5.out:Total Excitonic CCSD Energy =  -43.83857952810263
4.6.out:Total Excitonic CCSD Energy =  -43.83863931936915
4.7.out:Total Excitonic CCSD Energy =  -43.838675374594985
4.8.out:Total Excitonic CCSD Energy =  -43.83869204290733
4.9.out:Total Excitonic CCSD Energy =  -43.83869326644926
5.0.out:Total Excitonic CCSD Energy =  -43.83868255298337
5.1.out:Total Excitonic CCSD Energy =  -43.83866296946813
5.2.out:Total Excitonic CCSD Energy =  -43.838637149621874
5.3.out:Total Excitonic CCSD Energy =  -43.83860731102589
5.4.out:Total Excitonic CCSD Energy =  -43.83857527915674
5.5.out:Total Excitonic CCSD Energy =  -43.83854251683472
5.6.out:Total Excitonic CCSD Energy =  -43.83851015802553
5.7.out:Total Excitonic CCSD Energy =  -43.838479044938545
5.8.out:Total Excitonic CCSD Energy =  -43.83844976718979
5.9.out:Total Excitonic CCSD Energy =  -43.838422701604486
6.0.out:Total Excitonic CCSD Energy =  -43.8383980511826
6.1.out:Total Excitonic CCSD Energy =  -43.83837588185598
6.2.out:Total Excitonic CCSD Energy =  -43.83835615592192
6.3.out:Total Excitonic CCSD Energy =  -43.838338761383106
6.4.out:Total Excitonic CCSD Energy =  -43.83832353679214
6.5.out:Total Excitonic CCSD Energy =  -43.83831029153785
6.6.out:Total Excitonic CCSD Energy =  -43.838298821784505
6.7.out:Total Excitonic CCSD Energy =  -43.83828892246713
6.8.out:Total Excitonic CCSD Energy =  -43.83828039586015
6.9.out:Total Excitonic CCSD Energy =  -43.83827305728628
7.0.out:Total Excitonic CCSD Energy =  -43.83826673852245
7.1.out:Total Excitonic CCSD Energy =  -43.83826128942843
7.2.out:Total Excitonic CCSD Energy =  -43.83825657825785
7.3.out:Total Excitonic CCSD Energy =  -43.838252491043455
7.4.out:Total Excitonic CCSD Energy =  -43.8382489303774
7.5.out:Total Excitonic CCSD Energy =  -43.838245813839684
7.6.out:Total Excitonic CCSD Energy =  -43.83824307226609
7.7.out:Total Excitonic CCSD Energy =  -43.83824064799268
7.8.out:Total Excitonic CCSD Energy =  -43.83823849317398
7.9.out:Total Excitonic CCSD Energy =  -43.83823656823278
""")
