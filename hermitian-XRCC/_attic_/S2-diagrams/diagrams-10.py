
    						S2

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



    						H2

def s01s01h00(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.ccca0(i0,j0,p,t,v,q)
        #@ X.aa1(i1,j1,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.h00(p,q)
          X.ccca0pXXq_Hpq(i0,j0,t,v)
        @ X.aa1Xu_S0u(i1,j1,w,t)
        @ X.s01(v,w)
        )

def s01s01h10(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -(1/2) * (-1)**(X.n_i1 + X.P) * raw(
        #  X.cca0(i0,j0,t,v,q)
        #@ X.caa1(i1,j1,p,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.h10(p,q)
          X.cca0XXq_H1q(i0,j0,t,v,p)
        @ X.caa1XXu_S0u(i1,j1,p,w,t)
        @ X.s01(v,w)
        )

def s01s01h11(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.cc0(i0,j0,t,v)
        #@ X.caaa1(i1,j1,p,w,u,q)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.h11(p,q)
          X.cc0Xv_Sv1(i0,j0,t,w)
        @ X.caaa1pXXq_Hpq(i1,j1,w,u)
        @ X.s01(t,u)
        )

def s01s10h00(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ccaa0(i0,j0,p,t,w,q)
        #@ X.ca1(i1,j1,v,u)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.h00(p,q)
          X.ccaa0pXXq_Hpq(i0,j0,t,w)
        @ X.ca1Xu_S0u(i1,j1,v,t)
        @ X.s10(v,w)
        )

def s01s10h01(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * (-1)**(X.n_i1 + X.P) * raw(
        #  X.cca0(i0,j0,p,t,w)
        #@ X.caa1(i1,j1,v,u,q)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.h01(p,q)
          X.cca0XXw_S1w(i0,j0,p,t,v)
        @ X.caa1XXq_H0q(i1,j1,v,u,p)
        @ X.s01(t,u)
        )



    						V2

def s01s01v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.ccccaa0(i0,j0,p,q,t,v,s,r)
        #@ X.aa1(i1,j1,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.v0000(p,q,r,s)
          X.ccccaa0pqXXsr_Vpqrs(i0,j0,t,v)
        @ X.aa1Xu_S0u(i1,j1,w,t)
        @ X.s01(v,w)
        )

def s01s01v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_i1 + X.P) * raw(
        #  X.cccaa0(i0,j0,p,t,v,s,r)
        #@ X.caa1(i1,j1,q,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.v0100(p,q,r,s)
          X.cccaa0pXXsr_Vp1rs(i0,j0,t,v,q)
        @ X.caa1XXu_S0u(i1,j1,q,w,t)
        @ X.s01(v,w)
        )

def s01s01v0110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -2 * raw(
        #  X.ccca0(i0,j0,p,t,v,s)
        #@ X.caaa1(i1,j1,q,w,u,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.v0110(p,q,r,s)
          X.ccca0pXXs_Vp11s(i0,j0,t,v,q,r)
        @ X.caaa1XXuX_S0u(i1,j1,q,w,r,t)
        @ X.s01(v,w)
        )

def s01s01v1100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.ccaa0(i0,j0,t,v,s,r)
        #@ X.ccaa1(i1,j1,p,q,w,u)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.v1100(p,q,r,s)
          X.ccaa0XXsr_V11rs(i0,j0,t,v,p,q)
        @ X.ccaa1XXXu_S0u(i1,j1,p,q,w,t)
        @ X.s01(v,w)
        )

def s01s01v1110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (-1)**(X.n_i1 + X.P) * raw(
        #  X.cca0(i0,j0,t,v,s)
        #@ X.ccaaa1(i1,j1,p,q,w,u,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.v1110(p,q,r,s)
          X.cca0XvX_Sv1(i0,j0,t,s,w)
        @ X.ccaaa1pqXXr_Vpqr0(i1,j1,w,u,s)
        @ X.s01(t,u)
        )

def s01s01v1111(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return (1/2) * raw(
        #  X.cc0(i0,j0,t,v)
        #@ X.ccaaaa1(i1,j1,p,q,w,u,s,r)
        #@ X.s01(t,u)
        #@ X.s01(v,w)
        #@ X.v1111(p,q,r,s)
          X.cc0Xv_Sv1(i0,j0,t,w)
        @ X.ccaaaa1pqXXsr_Vpqrs(i1,j1,w,u)
        @ X.s01(t,u)
        )

def s01s10v0000(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.cccaaa0(i0,j0,p,q,t,w,s,r)
        #@ X.ca1(i1,j1,v,u)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.v0000(p,q,r,s)
          X.cccaaa0pqXXsr_Vpqrs(i0,j0,t,w)
        @ X.ca1Xu_s0u(i1,j1,v,t)
        @ X.s10(v,w)
        )

def s01s10v0010(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -2 * (-1)**(X.n_i1 + X.P) * raw(
        #  X.cccaa0(i0,j0,p,q,t,w,s)
        #@ X.caa1(i1,j1,v,u,r)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.v0010(p,q,r,s)
          X.cccaa0pqXXs_Vpq1s(i0,j0,t,w,r)
        @ X.caa1XuX_S0u(i1,j1,v,r,t)
        @ X.s10(v,w)
        )

def s01s10v0011(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -1 * raw(
        #  X.ccca0(i0,j0,p,q,t,w)
        #@ X.caaa1(i1,j1,v,u,s,r)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.v0011(p,q,r,s)
          X.ccca0XXXw_S1w(i0,j0,p,q,t,v)
        @ X.caaa1XXsr_V00rs(i1,j1,v,u,p,q)
        @ X.s01(t,u)
        )

def s01s10v0100(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return -2 * (-1)**(X.n_i1 + X.P) * raw(
        #  X.ccaaa0(i0,j0,p,t,w,s,r)
        #@ X.cca1(i1,j1,q,v,u)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.v0100(p,q,r,s)
          X.ccaaa0pXXsr_Vp1rs(i0,j0,t,w,q)
        @ X.cca1XXu_S0u(i1,j1,q,v,t)
        @ X.s10(v,w)
        )

def s01s10v0110(X, contract_last=False):
    if no_result(X, contract_last):  return []
    i0, i1, j0, j1 = state_indices(contract_last)
    return 4 * raw(
        #  X.ccaa0(i0,j0,p,t,w,s)
        #@ X.ccaa1(i1,j1,q,v,u,r)
        #@ X.s01(t,u)
        #@ X.s10(v,w)
        #@ X.v0110(p,q,r,s)
          X.ccaa0pXXs_Vp11s(i0,j0,t,w,q,r)
        @ X.ccaa1XXuX_S0u(i1,j1,q,v,r,t)
        @ X.s10(v,w)
        )
