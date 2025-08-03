#import numpy as np
from qode.math.tensornet import evaluate, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, a, b, c, d = 'ijklabcd'        # lower the number of quotes we need to type

def tdm_2p_pp(d_oo, t2, vec):
    ret = {}
    u1 = tl_tensor(tl.tensor(vec.ph.to_ndarray(), dtype=tl.float64))
    #u2 = tl_tensor(tl.tensor(vec.pphh.to_ndarray(), dtype=tl.float64))

    # zeroth order
    ret["vooo"] = d_oo(1,2) @ u1(3,0) - d_oo(1,3) @ u1(2,0)
    ret["ovoo"] = - d_oo(0,2) @ u1(3,1) + d_oo(0,3) @ u1(2,1)

    # first order
    i1_ov = evaluate(t2(i,0,a,1) @ u1(i,a))

    ret["oovo"] = - i1_ov(1,2) @ d_oo(0,3) + i1_ov(0,2) @ d_oo(1,3) + t2(0,1,a,2) @ u1(3,a)
    ret["ooov"] = i1_ov(1,3) @ d_oo(0,2) - i1_ov(0,3) @ d_oo(1,2) - t2(0,1,a,3) @ u1(2,a)
    ret["ovvv"] = - t2(i,0,2,3) @ u1(i,1)
    ret["vovv"] = t2(i,1,2,3) @ u1(i,0)
    """
    ret["vvoo"] = 0.5 * u2(2,3,0,1)
    """
    return ret



