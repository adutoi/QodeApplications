import pickle
import numpy
from qode.math.tensornet import raw, evaluate

class empty(object):
    pass

def norm(X):
    return numpy.linalg.norm(raw(X))

p,q,r,s = "pqrs"



Be = pickle.load(open("rho/Be631g-new-1e-6.pkl","rb"))

for i,j in Be.rho['c'][-1,0]:
    c     = Be.rho['c'    ][-1,0][i,j]
    cca   = Be.rho['cca'  ][-1,0][i,j]
    cccaa = Be.rho['cccaa'][-1,0][i,j]

    norm_c     = norm(c)
    norm_cca   = norm(cca)
    norm_cccaa = norm(cccaa)

    print(f"{i:2d} {j:2d}    {norm_c:.3f}  {norm_cca:.3f}  {norm_cccaa:.3f}", end="")

    ca_ = evaluate(c(p) @ cca(p,0,1))
    cca_ = c(0)@ca_(1,2) - c(1)@ca_(0,2)

    norm_diff = norm(cca - cca_)
    print(f"    {norm_diff/norm_cca:.3f}", end="")

    print()

    #norm_cccaa = norm(cccaa)
    #cccaa_ = -( cca(0,1,3)@ca_(2,4) - cca(0,1,4)@ca_(2,3) - cca(2,1,3)@ca_(0,4) + cca(2,1,4)@ca_(0,3) - cca(0,2,3)@ca_(1,4) + cca(0,2,4)@ca_(1,3) ) / 2
    #norm_diff = norm(cccaa - cccaa_)
    #print("cccaa", norm_cccaa, norm_diff)
    #cccaa_ = -( cca_(0,1,3)@ca_(2,4) - cca_(0,1,4)@ca_(2,3) - cca_(2,1,3)@ca_(0,4) + cca_(2,1,4)@ca_(0,3) - cca_(0,2,3)@ca_(1,4) + cca_(0,2,4)@ca_(1,3) ) / 2
    #norm_diff = norm(cccaa - cccaa_)
    #print("cccaa", norm_cccaa, norm_diff)

