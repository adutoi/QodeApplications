#import numpy as np
from qode.math.tensornet import evaluate, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, a, b, c, d = 'ijklabcd'        # lower the number of quotes we need to type

def tdm_1p_ip(mp2_diffdm_ov, t2, vec):
    ret = {}
    u1 = tl_tensor(tl.tensor(vec.h.to_ndarray(), dtype=tl.float64))
    u2 = tl_tensor(tl.tensor(vec.phh.to_ndarray(), dtype=tl.float64))
    f11 = evaluate(0.25 * t2(0,l,a,b) @ t2(1,l,a,b))
    f22 = evaluate(- 1/sqrt(2) * t2)

    # zeroth order
    ret["o"] = u1(0)# @ d_oo(j,0)

    # second order
    
    ret["o"] -= u1(j) @ f11(j,0)
    ret["v"] = u1(j) @ mp2_diffdm_ov(j,0) + u2(i,j,b) @ f22(i,j,b,0)
    
    return ret
    

def tdm_2p_ip(d_oo, t2, t2s, t2d, vec):
    ret = {}
    u1 = tl_tensor(tl.tensor(vec.h.to_ndarray(), dtype=tl.float64))
    u2 = tl_tensor(tl.tensor(vec.phh.to_ndarray(), dtype=tl.float64))


    ret["ooo"] = (d_oo(0,1) @ u1(2) - d_oo(0,2) @ u1(1)  # zeroth order
                  )

    ret["ovv"] = (u1(i) @ t2(i,0,1,2)  # first order
                  + u1(i) @ t2d(i,0,1,2))
    
    ret["voo"] = (sqrt(2) * u2(1,2,0)
                  + 1 * u1(2) @ t2s(1,0) - 1 * u1(1) @ t2s(2,0)  # generated
                  )
    ret["oov"] =  (- 1 * u1(1) @ t2s(0,2) + 1 * d_oo(1,0) @ u1(i) @ t2s(i,2)  # generated
                   + (1/sqrt(2)) * (+ 1 * u2(i,1,a) @ t2(i,0,a,2) + 1 * u2(j,1,a) @ t2(j,0,a,2) - 1 * d_oo(1,0) @ u2(i,j,a) @ t2(i,j,a,2)))  # generated
    ret["ovo"] =  (u1(2) @ t2s(0,1) - 1 * d_oo(2,0) @ u1(i) @ t2s(i,1)  # generated
                   + (1/sqrt(2)) * (- 1 * u2(i,2,a) @ t2(i,0,a,1) - 1 * u2(j,2,a) @ t2(j,0,a,1) + 1 * d_oo(2,0) @ u2(i,j,a) @ t2(i,j,a,1)))  # generated
    ret["vvv"] =  (1/sqrt(2)) * (- 1 * u2(i,j,0) @ t2(i,j,1,2))  # generated
    
    return ret


def tdm_3p_ip(d_oo, t2, vec):
    ret = {}
    u1 = tl_tensor(tl.tensor(vec.h.to_ndarray(), dtype=tl.float64))
    #u2 = tl_tensor(tl.tensor(vec.phh.to_ndarray(), dtype=tl.float64))
    i1_ovv = evaluate(u1(i) @ t2(i,0,1,2))
    #d_oo_sqrt2 = sqrt(2) * d_oo(0,1)

    # zeroth order
    ret["ooooo"] = (
        u1(2) @ d_oo(1,3) @ d_oo(0,4) - u1(2) @ d_oo(0,3) @ d_oo(1,4)
        + u1(4) @ d_oo(1,2) @ d_oo(0,3) - u1(4) @ d_oo(0,2) @ d_oo(1,3)
        + u1(3) @ d_oo(1,4) @ d_oo(0,2) - u1(3) @ d_oo(0,4) @ d_oo(1,2)
    )

    # first order
    ret["vvooo"] = u1(4) @ t2(2,3,0,1) + u1(3) @ t2(4,2,0,1) + u1(2) @ t2(3,4,0,1)
    ret["oovov"] = d_oo(0,3) @ i1_ovv(1,2,4) - d_oo(1,3) @ i1_ovv(0,2,4) + u1(3) @ t2(0,1,4,2)
    ret["oovvo"] = d_oo(1,4) @ i1_ovv(0,2,3) - d_oo(0,4) @ i1_ovv(1,2,3) + u1(4) @ t2(0,1,2,3)
    ret["ooovv"] = d_oo(1,2) @ i1_ovv(0,3,4) - d_oo(0,2) @ i1_ovv(1,3,4) + u1(2) @ t2(0,1,3,4)
    """
    ret["voooo"] = d_oo_sqrt2(1,4) @ u2(2,3,0) + d_oo_sqrt2(1,3) @ u2(4,2,0) + d_oo_sqrt2(1,2) @ u2(3,4,0)
    ret["ovooo"] = - d_oo_sqrt2(0,4) @ u2(2,3,1) - d_oo_sqrt2(0,3) @ u2(4,2,1) - d_oo_sqrt2(0,2) @ u2(3,4,1)
    """
    return ret

