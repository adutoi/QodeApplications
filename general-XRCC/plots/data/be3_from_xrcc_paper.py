#    (C) Copyright 2018 Anthony D. Dutoi
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#
from data.extract import X
from data import atom

def parse(n, fields):
	dist   = float(fields[0].split("_")[2])
	energy = float(fields[-1])
	return dist, energy-3*atom.energy

data = X(
parse,
"""\
ccsd_Be3_3.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8240617995
ccsd_Be3_3.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8269025498
ccsd_Be3_3.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8293065401
ccsd_Be3_3.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8313198939
ccsd_Be3_3.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8329871728
ccsd_Be3_3.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8343502615
ccsd_Be3_3.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8354501838
ccsd_Be3_3.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8363258895
ccsd_Be3_3.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8370137236
ccsd_Be3_3.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8375460798
ccsd_Be3_4.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.837951015
ccsd_Be3_4.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382526451
ccsd_Be3_4.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8384713958
ccsd_Be3_4.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8386243152
ccsd_Be3_4.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8387254712
ccsd_Be3_4.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8387870429
ccsd_Be3_4.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8388169392
ccsd_Be3_4.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8388234985
ccsd_Be3_4.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8388127978
ccsd_Be3_4.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8387897039
ccsd_Be3_5.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8387580953
ccsd_Be3_5.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8387210435
ccsd_Be3_5.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8386809595
ccsd_Be3_5.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8386397125
ccsd_Be3_5.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8385986081
ccsd_Be3_5.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8385589596
ccsd_Be3_5.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.838521369
ccsd_Be3_5.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8384863365
ccsd_Be3_5.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8384541598
ccsd_Be3_5.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8384249747
ccsd_Be3_6.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8383987909
ccsd_Be3_6.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.838375523
ccsd_Be3_6.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8383550173
ccsd_Be3_6.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8383370744
ccsd_Be3_6.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8383214681
ccsd_Be3_6.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8383079601
ccsd_Be3_6.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382963122
ccsd_Be3_6.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382862943
ccsd_Be3_6.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382776909
ccsd_Be3_6.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382703045
ccsd_Be3_7.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382639578
ccsd_Be3_7.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382584942
ccsd_Be3_7.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382537774
ccsd_Be3_7.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382496904
ccsd_Be3_7.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382461336
ccsd_Be3_7.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.838243023
ccsd_Be3_7.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382402885
ccsd_Be3_7.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382378718
ccsd_Be3_7.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382357247
ccsd_Be3_7.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382338072
ccsd_Be3_8.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382320866
ccsd_Be3_8.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382305357
ccsd_Be3_8.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382291322
ccsd_Be3_8.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382278574
ccsd_Be3_8.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382266957
ccsd_Be3_8.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382253389
ccsd_Be3_8.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382243692
ccsd_Be3_8.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382234791
ccsd_Be3_8.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382226607
ccsd_Be3_8.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.838221907
ccsd_Be3_9.0_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382212117
ccsd_Be3_9.1_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382205695
ccsd_Be3_9.2_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382199756
ccsd_Be3_9.3_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382194257
ccsd_Be3_9.4_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382189161
ccsd_Be3_9.5_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382184434
ccsd_Be3_9.6_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382180045
ccsd_Be3_9.7_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382175966
ccsd_Be3_9.8_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382172172
ccsd_Be3_9.9_path=..:..:..:..:dimer_H:6-31g:H:16-115-550:load=H:16-115-550:thresh=1e-6:4.5:u.pickle.log:Total Excitonic CCSD Energy =  -43.8382168639
""")
