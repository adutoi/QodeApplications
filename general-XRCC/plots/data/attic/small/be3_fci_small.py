# With the core frozen, this is the FCI energy as a function of distance

from data.extract import X
from data import atom_small as atom

def parse(n, fields):
	distance = float(fields[0].split("_")[1])
	energy   = float(fields[-1])
	return distance, energy-3*atom.energy

data = X(
parse,
"""\
Be3_3.0 .out:    Total CI energy =     -43.682849044631183
Be3_3.1 .out:    Total CI energy =     -43.687316879437724
Be3_3.2 .out:    Total CI energy =     -43.690981662944800
Be3_3.3 .out:    Total CI energy =     -43.693987452430058
Be3_3.4 .out:    Total CI energy =     -43.696452928309604
Be3_3.5 .out:    Total CI energy =     -43.675918229777643
Be3_3.6 .out:    Total CI energy =     -43.672187762973579
Be3_3.7 .out:    Total CI energy =     -43.667551833427211
Be3_3.8 .out:    Total CI energy =     -43.661909056184513
Be3_3.9 .out:    Total CI energy =     -43.655127521692904
Be3_4.0 .out:    Total CI energy =     -43.704270496811645
Be3_4.1 .out:    Total CI energy =     -43.704879247519912
Be3_4.2 .out:    Total CI energy =     -43.705374134739991
Be3_4.3 .out:    Total CI energy =     -43.705774882486850
Be3_4.4 .out:    Total CI energy =     -43.706097847761562
Be3_4.5 .out:    Total CI energy =     -43.703259246571335
Be3_4.6 .out:    Total CI energy =     -43.702719286347751
Be3_4.7 .out:    Total CI energy =     -43.702048559201437
Be3_4.8 .out:    Total CI energy =     -43.701232555114110
Be3_4.9 .out:    Total CI energy =     -43.700252870504528
Be3_5.0 .out:    Total CI energy =     -43.707027186005512
Be3_5.1 .out:    Total CI energy =     -43.707084468655772
Be3_5.2 .out:    Total CI energy =     -43.707127199150314
Be3_5.3 .out:    Total CI energy =     -43.707158553007716
Be3_5.4 .out:    Total CI energy =     -43.707181123449431
Be3_5.5 .out:    Total CI energy =     -43.706879396561391
Be3_5.6 .out:    Total CI energy =     -43.706804214645928
Be3_5.7 .out:    Total CI energy =     -43.706710953380941
Be3_5.8 .out:    Total CI energy =     -43.706597392462228
Be3_5.9 .out:    Total CI energy =     -43.706460777409283
Be3_6.0 .out:    Total CI energy =     -43.707223636031074
Be3_6.1 .out:    Total CI energy =     -43.707224178152899
Be3_6.2 .out:    Total CI energy =     -43.707224195838911
Be3_6.3 .out:    Total CI energy =     -43.707223910338115
Be3_6.4 .out:    Total CI energy =     -43.707223472031544
Be3_6.5 .out:    Total CI energy =     -43.707215423659420
Be3_6.6 .out:    Total CI energy =     -43.707211003256411
Be3_6.7 .out:    Total CI energy =     -43.707205093702747
Be3_6.8 .out:    Total CI energy =     -43.707197302739800
Be3_6.9 .out:    Total CI energy =     -43.707187151751377
Be3_7.0 .out:    Total CI energy =     -43.707221054628825
Be3_7.1 .out:    Total CI energy =     -43.707220838861680
Be3_7.2 .out:    Total CI energy =     -43.707220669683856
Be3_7.3 .out:    Total CI energy =     -43.707220539349350
Be3_7.4 .out:    Total CI energy =     -43.707220440510014
Be3_7.5 .out:    Total CI energy =     -43.707221429198356
Be3_7.6 .out:    Total CI energy =     -43.707221595902055
Be3_7.7 .out:    Total CI energy =     -43.707221774838686
Be3_7.8 .out:    Total CI energy =     -43.707221951089480
Be3_7.9 .out:    Total CI energy =     -43.707222103469086
""")
