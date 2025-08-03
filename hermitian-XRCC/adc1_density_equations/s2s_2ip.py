#import numpy as np
from qode.math.tensornet import evaluate, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, a, b, c, d = 'ijklabcd'        # lower the number of quotes we need to type

def s2s_2p_2ip(vec_left, vec_right):
    ret = {}
    ul1 = tl_tensor(tl.tensor(vec_left.h.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.phh.to_ndarray(), dtype=tl.float64))
    ur1 = tl_tensor(tl.tensor(vec_right.p.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.pph.to_ndarray(), dtype=tl.float64))

    # zeroth order
    ret["ov"] = ul1(0) @ ur1(1)
    ret["vo"] = - ul1(1) @ ur1(0)

    # first order
    """
    ret["oo"] = sqrt(2) * ul2(0,1,a) @ ur1(a)
    ret["vv"] = sqrt(2) * ul1(i) @ ur2(i,1,0)
    """
    return ret


def s2s_3p_2ip(d_oo, t2, vec_left, vec_right):
    ret = {}
    ul1 = tl_tensor(tl.tensor(vec_left.h.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.phh.to_ndarray(), dtype=tl.float64))
    ur1 = tl_tensor(tl.tensor(vec_right.p.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.pph.to_ndarray(), dtype=tl.float64))
    i1_oov = evaluate(ur1(a) @ t2(0,1,a,2))
    i2_ovv = evaluate(ul1(i) @ t2(i,0,1,2))
    #i3_oo = evaluate(sqrt(2) * ur1(a) @ ul2(0,1,a))
    #i4_vv = evaluate(sqrt(2) * ul1(i) @ ur2(i,0,1))
    #ur1_sqrt2 = evaluate(sqrt(2) * ur1(0))
    #ul1_sqrt2 = evaluate(sqrt(2) * ul1(0))

    # zeroth order
    ret["ovoo"] = d_oo(0,2) @ ul1(3) @ ur1(1) - d_oo(0,3) @ ul1(2) @ ur1(1)
    ret["oovo"] = d_oo(0,3) @ ul1(1) @ ur1(2) - d_oo(0,1) @ ul1(3) @ ur1(2)
    ret["ooov"] = d_oo(0,1) @ ul1(2) @ ur1(3) - d_oo(0,2) @ ul1(1) @ ur1(3)

    # first order
    ret["vooo"] = ul1(3) @ i1_oov(1,2,0) + ul1(1) @ i1_oov(2,3,0) + ul1(2) @ i1_oov(3,1,0)
    ret["ovvv"] = ur1(3) @ i2_ovv(0,1,2) + ur1(1) @ i2_ovv(0,2,3) + ur1(2) @ i2_ovv(0,3,1)
    """
    ret["oooo"] = d_oo(0,1) @ i3_oo(2,3) + d_oo(0,3) @ i3_oo(1,2) + d_oo(0,2) @ i3_oo(3,1)
    ret["voov"] = ur1_sqrt2(3) @ ul2(1,2,0)
    ret["vovo"] = ur1_sqrt2(2) @ ul2(3,1,0)
    ret["vvoo"] = ur1_sqrt2(1) @ ul2(2,3,0)
    ret["oovv"] = d_oo(0,1) @ i4_vv(3,2) + ul1_sqrt2(1) @ ur2(0,2,3)
    ret["ovov"] = d_oo(0,2) @ i4_vv(1,3) + ul1_sqrt2(2) @ ur2(0,3,1)
    ret["ovvo"] = d_oo(0,3) @ i4_vv(2,1) + ul1_sqrt2(3) @ ur2(0,1,2)
    """
    return ret
