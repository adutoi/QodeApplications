def s01s01u100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.ccca0(i0,j0,p,t,v,q)
        #@ X.aa1(i1,j1,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.u1_00(p,q)
          X.ccca0pXXq_U1pq(i0,j0,t,v)
        @ X.aa1Xu_S0u(i1,j1,w,t)
        @ X.s01(v,w)
        )

def s01s01u110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -(1/2) * (-1)**(X.n_i1 + X.P) * raw(
        #  X.cca0(i0,j0,t,v,q)
        #@ X.caa1(i1,j1,p,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.u1_10(p,q)
          X.cca0XXq_U11q(i0,j0,t,v,p)
        @ X.caa1XXu_S0u(i1,j1,p,w,t)
        @ X.s01(v,w)
        )

def s01s01u111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.cc0(i0,j0,t,v)
        #@ X.caaa1(i1,j1,p,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.u1_11(p,q)
          X.cc0Xv_Sv1(i0,j0,t,w)
        @ X.caaa1pXXq_U1pq(i1,j1,w,u)
        @ X.s01(t,u)
        )

def s01s10u100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ccaa0(i0,j0,p,t,w,q)
        #@ X.ca1(i1,j1,v,u)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.u1_00(p,q)
          X.ccaa0pXXq_U1pq(i0,j0,t,w)
        @ X.ca1Xu_S0u(i1,j1,v,t)
        @ X.s10(v,w)
        )

def s01s10u101(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * (-1)**(X.n_i1 + X.P) * raw(
        #  X.cca0(i0,j0,p,t,w)
        #@ X.caa1(i1,j1,v,u,q)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.u1_01(p,q)
          X.cca0XXw_S1w(i0,j0,p,t,v)
        @ X.caa1XXq_U10q(i1,j1,v,u,p)
        @ X.s01(t,u)
        )
