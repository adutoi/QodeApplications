def s01s01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.cc0(i0,j0,p,r)
        #@ X.aa1(i1,j1,s,q)
        #@ X.s01(p,q)
        #@ X.s01(r,s)
          X.cc0Xr_Sr1(i0,j0,p,s)
        @ X.aa1Xq_S0q(i1,j1,s,p)
        )

def s01s10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ca0(i0,j0,p,s)
        #@ X.ca1(i1,j1,r,q)
        #@ X.s01(p,q)
        #@ X.s10(r,s)
          X.ca0Xs_S1s(i0,j0,p,r)
        @ X.ca1Xq_S0q(i1,j1,r,p)
        )
