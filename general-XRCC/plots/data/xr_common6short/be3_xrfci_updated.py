from data.extract import X
from data import atom

def parse(n, fields):
        dist   = float(fields[0].split(".out")[0])
        energy = float(fields[-1])
        return dist, energy-3*atom.energy

data = X( 
parse,
"""\
3.0.out:Total Excitonic FCI  Energy =  -43.84425754871073
3.1.out:Total Excitonic FCI  Energy =  -43.841725168042636
3.2.out:Total Excitonic FCI  Energy =  -43.84013681483737
3.3.out:Total Excitonic FCI  Energy =  -43.839158515332855
3.4.out:Total Excitonic FCI  Energy =  -43.8385800304738
3.5.out:Total Excitonic FCI  Energy =  -43.838266054488265
3.6.out:Total Excitonic FCI  Energy =  -43.83812661827684
3.7.out:Total Excitonic FCI  Energy =  -43.83809987603301
3.8.out:Total Excitonic FCI  Energy =  -43.8381422952389
3.9.out:Total Excitonic FCI  Energy =  -43.83822298976624
4.0.out:Total Excitonic FCI  Energy =  -43.83832026280006
4.1.out:Total Excitonic FCI  Energy =  -43.83841930671858
4.2.out:Total Excitonic FCI  Energy =  -43.83851052805074
4.3.out:Total Excitonic FCI  Energy =  -43.83858824622791
4.4.out:Total Excitonic FCI  Energy =  -43.83864965260643
4.5.out:Total Excitonic FCI  Energy =  -43.83869397675076
4.6.out:Total Excitonic FCI  Energy =  -43.83872182931785
4.7.out:Total Excitonic FCI  Energy =  -43.83873469681021
4.8.out:Total Excitonic FCI  Energy =  -43.83873456399518
4.9.out:Total Excitonic FCI  Energy =  -43.838723640030395
5.0.out:Total Excitonic FCI  Energy =  -43.83870416588753
5.1.out:Total Excitonic FCI  Energy =  -43.83867828364192
5.2.out:Total Excitonic FCI  Energy =  -43.83864795189105
5.3.out:Total Excitonic FCI  Energy =  -43.83861489529982
5.4.out:Total Excitonic FCI  Energy =  -43.83858057944803
5.5.out:Total Excitonic FCI  Energy =  -43.8385462045809
5.6.out:Total Excitonic FCI  Energy =  -43.83851271347064
5.7.out:Total Excitonic FCI  Energy =  -43.838480809586
5.8.out:Total Excitonic FCI  Energy =  -43.838450982358665
5.9.out:Total Excitonic FCI  Energy =  -43.83842353671765
6.0.out:Total Excitonic FCI  Energy =  -43.8383986244347
6.1.out:Total Excitonic FCI  Energy =  -43.83837627522204
6.2.out:Total Excitonic FCI  Energy =  -43.83835642596675
6.3.out:Total Excitonic FCI  Energy =  -43.83833894697641
6.4.out:Total Excitonic FCI  Energy =  -43.838323664564285
6.5.out:Total Excitonic FCI  Energy =  -43.838310379699834
6.6.out:Total Excitonic FCI  Energy =  -43.83829888278031
6.7.out:Total Excitonic FCI  Energy =  -43.83828896480175
6.8.out:Total Excitonic FCI  Energy =  -43.83828042535206
6.9.out:Total Excitonic FCI  Energy =  -43.83827307792211
7.0.out:Total Excitonic FCI  Energy =  -43.838266753037686
7.1.out:Total Excitonic FCI  Energy =  -43.83826129970398
7.2.out:Total Excitonic FCI  Energy =  -43.83825658558902
7.3.out:Total Excitonic FCI  Energy =  -43.83825249632315
7.4.out:Total Excitonic FCI  Energy =  -43.8382489342223
7.5.out:Total Excitonic FCI  Energy =  -43.83824581667632
7.6.out:Total Excitonic FCI  Energy =  -43.83824307438944
7.7.out:Total Excitonic FCI  Energy =  -43.83824064960764
7.8.out:Total Excitonic FCI  Energy =  -43.83823849442278
7.9.out:Total Excitonic FCI  Energy =  -43.83823656921475
""")
