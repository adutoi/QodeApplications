from util import *


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

