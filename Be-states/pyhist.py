class empty(object):  pass
from diffs import *
numpy.set_printoptions(precision=4)

diffs("cca",0,1)
diff("cca",0,1,0,0)

Acca = AA("cca",0,1,0,0)
Bcca = BB("cca",0,1,0,0)
Dcca = Acca - Bcca

numpy.linalg.norm(Dcca)
numpy.linalg.norm(Dcca[:,:,0])
numpy.linalg.norm(Dcca[:,:,1])
numpy.linalg.norm(Dcca[:,:,2])
numpy.linalg.norm(Dcca[:,:,3])
numpy.linalg.norm(Dcca[:,:,4])
numpy.linalg.norm(Dcca[:,:,5])
numpy.linalg.norm(Dcca[:,:,6])
numpy.linalg.norm(Dcca[:,:,7])
numpy.linalg.norm(Dcca[:,:,8])
numpy.linalg.norm(Dcca[:,:,9])
numpy.linalg.norm(Dcca[:,:,10])
numpy.linalg.norm(Dcca[:,:,11])
numpy.linalg.norm(Dcca[:,:,12])
numpy.linalg.norm(Dcca[:,:,13])
numpy.linalg.norm(Dcca[:,:,14])
numpy.linalg.norm(Dcca[:,:,15])
numpy.linalg.norm(Dcca[:,:,16])
numpy.linalg.norm(Dcca[:,:,17])

sum(numpy.linalg.norm(Dcca[:,:,i])**2 for i in [0,1,5,9])**0.5

Dcca[:,:,0]
Acca[:,:,0]
Bcca[:,:,0]

import readline
readline.write_history_file("pyhist2.py")
