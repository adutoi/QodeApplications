from data.extract import X
from data import atom

def parse(n, fields):
	dist   = float(fields[0].split("_")[2])
	energy = float(fields[-1])
	return dist, energy-2*atom.energy

data = X(
parse,
"""\
ccsd_Be2_3.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2212632091
ccsd_Be2_3.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2221035908
ccsd_Be2_3.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2228217627
ccsd_Be2_3.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2234314909
ccsd_Be2_3.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2239447397
ccsd_Be2_3.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2243715912
ccsd_Be2_3.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2247218455
ccsd_Be2_3.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2250049775
ccsd_Be2_3.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2252302429
ccsd_Be2_3.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254061927
ccsd_Be2_4.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255405079
ccsd_Be2_4.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2256401178
ccsd_Be2_4.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2257111801
ccsd_Be2_4.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2257590673
ccsd_Be2_4.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2257883967
ccsd_Be2_4.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2258034218
ccsd_Be2_4.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2258067161
ccsd_Be2_4.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2258014087
ccsd_Be2_4.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2257897596
ccsd_Be2_4.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2257736192
ccsd_Be2_5.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2257544897
ccsd_Be2_5.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2257335786
ccsd_Be2_5.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2257118449
ccsd_Be2_5.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2256900377
ccsd_Be2_5.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2256686709
ccsd_Be2_5.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.225648297
ccsd_Be2_5.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2256291401
ccsd_Be2_5.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.225611394
ccsd_Be2_5.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255951662
ccsd_Be2_5.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255804953
ccsd_Be2_6.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255673655
ccsd_Be2_6.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255557195
ccsd_Be2_6.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255454707
ccsd_Be2_6.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.225536513
ccsd_Be2_6.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255287288
ccsd_Be2_6.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255219963
ccsd_Be2_6.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255161946
ccsd_Be2_6.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255112076
ccsd_Be2_6.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.225506927
ccsd_Be2_6.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2255032538
ccsd_Be2_7.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.225500099
ccsd_Be2_7.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254973844
ccsd_Be2_7.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.225495042
ccsd_Be2_7.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254930131
ccsd_Be2_7.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254912481
ccsd_Be2_7.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254897052
ccsd_Be2_7.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254883493
ccsd_Be2_7.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254871514
ccsd_Be2_7.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254860874
ccsd_Be2_7.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254851375
ccsd_Be2_8.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254842854
ccsd_Be2_8.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254835175
ccsd_Be2_8.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254828226
ccsd_Be2_8.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254821917
ccsd_Be2_8.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254816168
ccsd_Be2_8.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254810916
ccsd_Be2_8.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254806106
ccsd_Be2_8.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254801691
ccsd_Be2_8.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.225479763
ccsd_Be2_8.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.225479389
ccsd_Be2_9.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254790438
ccsd_Be2_9.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.225478725
ccsd_Be2_9.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254784301
ccsd_Be2_9.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.225478157
ccsd_Be2_9.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254779038
ccsd_Be2_9.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254776689
ccsd_Be2_9.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254774508
ccsd_Be2_9.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254772479
ccsd_Be2_9.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254770592
ccsd_Be2_9.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -29.2254768835
""")
