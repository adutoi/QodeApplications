#    (C) Copyright 2018 Anthony D. Dutoi
# 
#    This file is part of QodeApplications.
# 
#    QodeApplications is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    QodeApplications is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with QodeApplications.  If not, see <http://www.gnu.org/licenses/>.
#
from data.extract import X
from data import atom

def parse(n, fields):
	subfields = float(fields[0].split("_")[3])
	energy    = float(fields[-1])
	return subfields, energy-3*atom.energy

data = X(
parse,
"""\
be3_fci_6-31g_3.000_A.out:* ROOT 1 CI total energy = -43.8326388130099
be3_fci_6-31g_3.100_A.out:* ROOT 1 CI total energy = -43.8334925400375
be3_fci_6-31g_3.200_A.out:* ROOT 1 CI total energy = -43.8343462624985
be3_fci_6-31g_3.300_A.out:* ROOT 1 CI total energy = -43.8351525440967
be3_fci_6-31g_3.400_A.out:* ROOT 1 CI total energy = -43.8358834060515
be3_fci_6-31g_3.500_A.out:* ROOT 1 CI total energy = -43.8365250824033
be3_fci_6-31g_3.600_A.out:* ROOT 1 CI total energy = -43.8370735774139
be3_fci_6-31g_3.700_A.out:* ROOT 1 CI total energy = -43.8375312615592
be3_fci_6-31g_3.800_A.out:* ROOT 1 CI total energy = -43.8379043863613
be3_fci_6-31g_3.900_A.out:* ROOT 1 CI total energy = -43.8382013244332
be3_fci_6-31g_4.000_A.out:* ROOT 1 CI total energy = -43.8384313571287
be3_fci_6-31g_4.100_A.out:* ROOT 1 CI total energy = -43.8386038674818
be3_fci_6-31g_4.200_A.out:* ROOT 1 CI total energy = -43.8387278290490
be3_fci_6-31g_4.300_A.out:* ROOT 1 CI total energy = -43.8388115074535
be3_fci_6-31g_4.400_A.out:* ROOT 1 CI total energy = -43.8388623114171
be3_fci_6-31g_4.500_A.out:* ROOT 1 CI total energy = -43.8388867453853
be3_fci_6-31g_4.600_A.out:* ROOT 1 CI total energy = -43.8388904277534
be3_fci_6-31g_4.700_A.out:* ROOT 1 CI total energy = -43.8388781481242
be3_fci_6-31g_4.800_A.out:* ROOT 1 CI total energy = -43.8388539445252
be3_fci_6-31g_4.900_A.out:* ROOT 1 CI total energy = -43.8388211875222
be3_fci_6-31g_5.000_A.out:* ROOT 1 CI total energy = -43.8387826628908
be3_fci_6-31g_5.100_A.out:* ROOT 1 CI total energy = -43.8387406481394
be3_fci_6-31g_5.200_A.out:* ROOT 1 CI total energy = -43.8386969808323
be3_fci_6-31g_5.300_A.out:* ROOT 1 CI total energy = -43.8386531184558
be3_fci_6-31g_5.400_A.out:* ROOT 1 CI total energy = -43.8386101906316
be3_fci_6-31g_5.500_A.out:* ROOT 1 CI total energy = -43.8385690449579
be3_fci_6-31g_5.600_A.out:* ROOT 1 CI total energy = -43.8385302878188
be3_fci_6-31g_5.700_A.out:* ROOT 1 CI total energy = -43.8384943212817
be3_fci_6-31g_5.800_A.out:* ROOT 1 CI total energy = -43.8384613768673
be3_fci_6-31g_5.900_A.out:* ROOT 1 CI total energy = -43.8384315466255
be3_fci_6-31g_6.000_A.out:* ROOT 1 CI total energy = -43.8384048116460
be3_fci_6-31g_6.100_A.out:* ROOT 1 CI total energy = -43.8383810679409
be3_fci_6-31g_6.200_A.out:* ROOT 1 CI total energy = -43.8383601495397
be3_fci_6-31g_6.300_A.out:* ROOT 1 CI total energy = -43.8383418486310
be3_fci_6-31g_6.400_A.out:* ROOT 1 CI total energy = -43.8383259326479
be3_fci_6-31g_6.500_A.out:* ROOT 1 CI total energy = -43.8383121582980
be3_fci_6-31g_6.600_A.out:* ROOT 1 CI total energy = -43.8383002826397
be3_fci_6-31g_6.700_A.out:* ROOT 1 CI total energy = -43.8382900714178
be3_fci_6-31g_6.800_A.out:* ROOT 1 CI total energy = -43.8382813049468
be3_fci_6-31g_6.900_A.out:* ROOT 1 CI total energy = -43.8382737818822
be3_fci_6-31g_7.000_A.out:* ROOT 1 CI total energy = -43.8382673212471
be3_fci_6-31g_7.100_A.out:* ROOT 1 CI total energy = -43.8382617630775
be3_fci_6-31g_7.200_A.out:* ROOT 1 CI total energy = -43.8382569680279
be3_fci_6-31g_7.300_A.out:* ROOT 1 CI total energy = -43.8382528162418
be3_fci_6-31g_7.400_A.out:* ROOT 1 CI total energy = -43.8382492057540
be3_fci_6-31g_7.500_A.out:* ROOT 1 CI total energy = -43.8382460506249
be3_fci_6-31g_7.600_A.out:* ROOT 1 CI total energy = -43.8382432789847
be3_fci_6-31g_7.700_A.out:* ROOT 1 CI total energy = -43.8382408311011
be3_fci_6-31g_7.800_A.out:* ROOT 1 CI total energy = -43.8382386575558
be3_fci_6-31g_7.900_A.out:* ROOT 1 CI total energy = -43.8382367175841
be3_fci_6-31g_8.000_A.out:* ROOT 1 CI total energy = -43.8382349775992
be3_fci_6-31g_8.100_A.out:* ROOT 1 CI total energy = -43.8382334099150
be3_fci_6-31g_8.200_A.out:* ROOT 1 CI total energy = -43.8382319916602
be3_fci_6-31g_8.300_A.out:* ROOT 1 CI total energy = -43.8382307038682
be3_fci_6-31g_8.400_A.out:* ROOT 1 CI total energy = -43.8382295307296
be3_fci_6-31g_8.500_A.out:* ROOT 1 CI total energy = -43.8382284589805
be3_fci_6-31g_8.600_A.out:* ROOT 1 CI total energy = -43.8382274774117
be3_fci_6-31g_8.700_A.out:* ROOT 1 CI total energy = -43.8382265764742
be3_fci_6-31g_8.800_A.out:* ROOT 1 CI total energy = -43.8382257479675
be3_fci_6-31g_8.900_A.out:* ROOT 1 CI total energy = -43.8382249847916
be3_fci_6-31g_9.000_A.out:* ROOT 1 CI total energy = -43.8382242807522
be3_fci_6-31g_9.100_A.out:* ROOT 1 CI total energy = -43.8382236304059
be3_fci_6-31g_9.200_A.out:* ROOT 1 CI total energy = -43.8382230289388
be3_fci_6-31g_9.300_A.out:* ROOT 1 CI total energy = -43.8382224720691
be3_fci_6-31g_9.400_A.out:* ROOT 1 CI total energy = -43.8382219559694
be3_fci_6-31g_9.500_A.out:* ROOT 1 CI total energy = -43.8382214772035
be3_fci_6-31g_9.600_A.out:* ROOT 1 CI total energy = -43.8382210326762
be3_fci_6-31g_9.700_A.out:* ROOT 1 CI total energy = -43.8382206195904
be3_fci_6-31g_9.800_A.out:* ROOT 1 CI total energy = -43.8382202354117
be3_fci_6-31g_9.900_A.out:* ROOT 1 CI total energy = -43.8382198778390
be3_fci_6-31g_10.000_A.out:* ROOT 1 CI total energy = -43.8382195447791
""")
