#import numpy as np
from qode.math.tensornet import evaluate, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, a, b, c, d = 'ijklabcd'        # lower the number of quotes we need to type

def tdm_1p_ea(mp2_diffdm_ov, t2, vec):
    ret = {}
    u1 = tl_tensor(tl.tensor(vec.p.to_ndarray(), dtype=tl.float64))
    u2 = tl_tensor(tl.tensor(vec.pph.to_ndarray(), dtype=tl.float64))
    f11 = evaluate(0.25 * t2(i,j,1,c) @ t2(i,j,0,c))
    f22 = evaluate(1/sqrt(2) * t2)

    # zeroth order
    ret["v"] = u1(0)# @ d_vv(b,0)

    # second order
    
    ret["v"] -= u1(b) @ f11(b,0)
    ret["o"] = - mp2_diffdm_ov(0,a) @ u1(a) + f22(0,j,a,b) @ u2(j,a,b)
    
    return ret
    

def tdm_2p_ea(d_oo, t2, t2s, t2d, vec):
    ret = {}
    u1 = tl_tensor(tl.tensor(vec.p.to_ndarray(), dtype=tl.float64))
    u2 = tl_tensor(tl.tensor(vec.pph.to_ndarray(), dtype=tl.float64))
    z = 1/sqrt(2)


    ret["voo"] = u1(0) @ d_oo(1,2)  # zeroth order
                  
    ret["ovo"] = u1(1) @ d_oo(0,2)  # zeroth order
                  

    ret["oov"] = u1(a) @ t2(0,1,a,2)\
                  + u1(a) @ t2d(0,1,a,2)
    
    ret["vvo"] = sqrt(2) * u2(2,0,1)\
                  + 1 * u1(0) @ t2s(2,1) - 1 * u1(1) @ t2s(2,0)  # generated
                  

    ret["ooo"] =  d_oo(2,0) @ u1(a) @ t2s(1,a) - 1 * d_oo(2,1) @ u1(a) @ t2s(0,a)\
                   - z * u2(2,a,b) @ t2(0,1,a,b) + z * d_oo(2,0) @ u2(i,a,b) @ t2(i,1,a,b) - z * d_oo(2,1) @ u2(i,a,b) @ t2(i,0,a,b)  # generated
                  
    ret["ovv"] =  - 1 * u1(1) @ t2s(0,2)\
                   - z * u2(i,a,1) @ t2(i,0,a,2) - z * u2(i,b,1) @ t2(i,0,b,2)  # generated
                   
    ret["vov"] =  u1(0) @ t2s(1,2)\
                   + z * u2(i,a,0) @ t2(i,1,a,2) + z * u2(i,b,0) @ t2(i,1,b,2)  # generated
                   
    
    return ret


def tdm_3p_ea(d_oo, t2, vec):
    ret = {}
    u1 = tl_tensor(tl.tensor(vec.p.to_ndarray(), dtype=tl.float64))
    #u2 = tl_tensor(tl.tensor(vec.pph.to_ndarray(), dtype=tl.float64))
    i1_oov = evaluate(u1(a) @ t2(0,1,a,2))
    #d_oo_sqrt2 = sqrt(2) * d_oo(0,1)

    # zeroth order
    ret["voooo"] = u1(0) @ d_oo(1,4) @ d_oo(2,3) - u1(0) @ d_oo(1,3) @ d_oo(2,4)
    ret["ovooo"] = u1(1) @ d_oo(2,4) @ d_oo(0,3) - u1(1) @ d_oo(2,3) @ d_oo(0,4)
    ret["oovoo"] = u1(2) @ d_oo(0,4) @ d_oo(1,3) - u1(2) @ d_oo(0,3) @ d_oo(1,4)

    # first order
    ret["vvvoo"] = u1(2) @ t2(3,4,0,1) + u1(1) @ t2(3,4,2,0) + u1(0) @ t2(3,4,1,2)
    ret["ooovo"] = d_oo(1,4) @ i1_oov(0,2,3) - d_oo(2,4) @ i1_oov(0,1,3) - d_oo(0,4) @ i1_oov(1,2,3)
    ret["oooov"] = - d_oo(1,3) @ i1_oov(0,2,4) + d_oo(2,3) @ i1_oov(0,1,4) + d_oo(0,3) @ i1_oov(1,2,4)
    """
    ret["vvooo"] = d_oo_sqrt2(2,3) @ u2(4,0,1) - d_oo_sqrt2(2,4) @ u2(3,0,1)
    ret["vovoo"] = d_oo_sqrt2(1,3) @ u2(4,2,0) - d_oo_sqrt2(1,4) @ u2(3,2,0)
    ret["ovvoo"] = d_oo_sqrt2(0,3) @ u2(4,1,2) - d_oo_sqrt2(0,4) @ u2(3,1,2)
    """
    return ret

