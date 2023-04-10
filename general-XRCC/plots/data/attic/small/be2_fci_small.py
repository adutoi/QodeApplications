# With the core frozen, this is the FCI energy as a function of distance

from data.extract import X
from data import atom_small as atom

def parse(n, fields):
	distance = float(fields[0].split("_")[1])
	energy   = float(fields[-1])
	return distance, energy-2*atom.energy

data = X(
parse,
"""\
Be2_3.0 .out:    Total CI energy =     -29.126124400778984
Be2_3.1 .out:    Total CI energy =     -29.128311659216873
Be2_3.2 .out:    Total CI energy =     -29.130108763815031
Be2_3.3 .out:    Total CI energy =     -29.131585481701062
Be2_3.4 .out:    Total CI energy =     -29.132799210428271
Be2_3.5 .out:    Total CI energy =     -29.133796994895693
Be2_3.6 .out:    Total CI energy =     -29.134617292820138
Be2_3.7 .out:    Total CI energy =     -29.135291501226028
Be2_3.8 .out:    Total CI energy =     -29.135845255311185
Be2_3.9 .out:    Total CI energy =     -29.136299515527313
Be2_4.0 .out:    Total CI energy =     -29.136671463675995
Be2_4.1 .out:    Total CI energy =     -29.136975232044133
Be2_4.2 .out:    Total CI energy =     -29.137222490552482
Be2_4.3 .out:    Total CI energy =     -29.137422915903205
Be2_4.4 .out:    Total CI energy =     -29.137584564455590
Be2_4.5 .out:    Total CI energy =     -29.137714167688685
Be2_4.6 .out:    Total CI energy =     -29.137817366133657
Be2_4.7 .out:    Total CI energy =     -29.137898894877200
Be2_4.8 .out:    Total CI energy =     -29.137962731294220
Be2_4.9 .out:    Total CI energy =     -29.138012213587039
Be2_5.0 .out:    Total CI energy =     -29.138050136955460
Be2_5.1 .out:    Total CI energy =     -29.138078832756392
Be2_5.2 .out:    Total CI energy =     -29.138100234777419
Be2_5.3 .out:    Total CI energy =     -29.138115935718726
Be2_5.4 .out:    Total CI energy =     -29.138127236133077
Be2_5.5 .out:    Total CI energy =     -29.138135187395950
Be2_5.6 .out:    Total CI energy =     -29.138140629756307
Be2_5.7 .out:    Total CI energy =     -29.138144226142959
Be2_5.8 .out:    Total CI energy =     -29.138146492144987
Be2_5.9 .out:    Total CI energy =     -29.138147822434814
Be2_6.0 .out:    Total CI energy =     -29.138148513828064
Be2_6.1 .out:    Total CI energy =     -29.138148785154087
Be2_6.2 .out:    Total CI energy =     -29.138148794127634
Be2_6.3 .out:    Total CI energy =     -29.138148651438808
Be2_6.4 .out:    Total CI energy =     -29.138148432313304
Be2_6.5 .out:    Total CI energy =     -29.138148185817712
Be2_6.6 .out:    Total CI energy =     -29.138147942202004
Be2_6.7 .out:    Total CI energy =     -29.138147718569751
Be2_6.8 .out:    Total CI energy =     -29.138147523158306
Be2_6.9 .out:    Total CI energy =     -29.138147358490414
Be2_7.0 .out:    Total CI energy =     -29.138147223629876
Be2_7.1 .out:    Total CI energy =     -29.138147115746207
Be2_7.2 .out:    Total CI energy =     -29.138147031157214
Be2_7.3 .out:    Total CI energy =     -29.138146965989915
Be2_7.4 .out:    Total CI energy =     -29.138146916570197
Be2_7.5 .out:    Total CI energy =     -29.138146879627136
Be2_7.6 .out:    Total CI energy =     -29.138146852374373
Be2_7.7 .out:    Total CI energy =     -29.138146832517339
Be2_7.8 .out:    Total CI energy =     -29.138146818216473
Be2_7.9 .out:    Total CI energy =     -29.138146808030353
""")
