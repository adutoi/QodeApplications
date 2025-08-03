#import numpy as np
from qode.math.tensornet import evaluate, tl_tensor
import tensorly as tl
from math import sqrt

i, j, k, l, a, b, c, d = 'ijklabcd'        # lower the number of quotes we need to type

def s2s_2p_2ea(vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.h.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.phh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.p.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.pph.to_ndarray(), dtype=tl.float64))

    # zeroth order
    ret["vo"] = ul1(0) @ ur1(1)
    ret["ov"] = - ul1(1) @ ur1(0)

    # first order
    """
    ret["vv"] = sqrt(2) * ul2(i,0,1) @ ur1(i)
    ret["oo"] = sqrt(2) * ul1(a) @ ur2(1,0,a)
    """
    return ret


def s2s_3p_2ea(d_oo, t2, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.h.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.phh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.p.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.pph.to_ndarray(), dtype=tl.float64))
    i1_ovv = evaluate(ur1(i) @ t2(i,0,1,2))
    i2_oov = evaluate(ul1(a) @ t2(0,1,a,2))
    #i3_vv = evaluate(sqrt(2) * ur1(i) @ ul2(i,0,1))
    #i4_oo = evaluate(sqrt(2) * ul1(a) @ ur2(0,1,a))
    #ur1_sqrt2 = evaluate(sqrt(2) * ur1(0))
    #ul1_sqrt2 = evaluate(sqrt(2) * ul1(0))

    # zeroth order
    ret["vooo"] = d_oo(2,3) @ ul1(0) @ ur1(1) - d_oo(1,3) @ ul1(0) @ ur1(2)
    ret["ovoo"] = - d_oo(2,3) @ ul1(1) @ ur1(0) + d_oo(0,3) @ ul1(1) @ ur1(2)
    ret["oovo"] = d_oo(1,3) @ ul1(2) @ ur1(0) - d_oo(0,3) @ ul1(2) @ ur1(1)

    # first order
    
    ret["vvvo"] = - ul1(0) @ i1_ovv(3,1,2) - ul1(2) @ i1_ovv(3,0,1) - ul1(1) @ i1_ovv(3,2,0)
    ret["ooov"] = - ur1(0) @ i2_oov(1,2,3) - ur1(2) @ i2_oov(0,1,3) - ur1(1) @ i2_oov(2,0,3)
    """
    ret["vvoo"] = d_oo(2,3) @ i3_vv(0,1) + ur1_sqrt2(2) @ ul2(3,1,0)
    ret["vovo"] = d_oo(1,3) @ i3_vv(2,0) + ur1_sqrt2(1) @ ul2(3,0,2)
    ret["ovvo"] = d_oo(0,3) @ i3_vv(1,2) + ur1_sqrt2(0) @ ul2(3,2,1)
    ret["oooo"] = d_oo(1,3) @ i4_oo(0,2) + d_oo(2,3) @ i4_oo(1,0) + d_oo(0,3) @ i4_oo(2,1)
    ret["voov"] = ul1_sqrt2(0) @ ur2(2,1,3)
    ret["ovov"] = ul1_sqrt2(1) @ ur2(0,2,3)
    ret["oovv"] = ul1_sqrt2(2) @ ur2(1,0,3)
    """
    return ret
