#import numpy as np
from qode.math.tensornet import evaluate, scalar_value, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, a, b, c, d = 'ijklabcd'        # lower the number of quotes we need to type

def s2s_2p_ea_0(d_oo, t2, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.p.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.pph.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.p.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.pph.to_ndarray(), dtype=tl.float64))
    #i1 = scalar_value(ul1(a) @ ur1(a))
    i2_oov_r = evaluate(ur1(b) @ t2(0,1,b,2))
    i3_oov_l = evaluate(ul1(a) @ t2(0,1,a,2))
    #ur1_sqrt2 = evaluate(sqrt(2) * ur1)
    #ul1_sqrt2 = evaluate(sqrt(2) * ul1)
    #i4_ov = evaluate(ul2(0,1,b) @ ur1_sqrt2(b))
    #i5_ov = evaluate(ul1_sqrt2(a) @ ur2(0,a,1))

    # zeroth order
    #ret["oooo"] = (
    #    i1 * d_oo(0,3) @ d_oo(1,2)
    #    - i1 * d_oo(1,3) @ d_oo(0,2)
    #)
    ret["voov"] = ul1(0) @ ur1(3) @ d_oo(1,2)
    ret["ovov"] = - ul1(1) @ ur1(3) @ d_oo(0,2)
    ret["vovo"] = - ul1(0) @ ur1(2) @ d_oo(1,3)
    ret["ovvo"] = ul1(1) @ ur1(2) @ d_oo(0,3)

    # first order
    ret["vvoo"] = ul1(1) @ i2_oov_r(2,3,0) - ul1(0) @ i2_oov_r(2,3,1)
    ret["oovv"] = ur1(3) @ i3_oov_l(0,1,2) - ur1(2) @ i3_oov_l(0,1,3)
    """
    ret["vvov"] = ul2(2,0,1) @ ur1_sqrt2(3)
    ret["vvvo"] = - ul2(3,0,1) @ ur1_sqrt2(2)
    ret["vovv"] = ul1_sqrt2(0) @ ur2(1,3,2)
    ret["ovvv"] = ul1_sqrt2(1) @ ur2(0,3,2)
    ret["vooo"] = - i4_ov(3,0) @ d_oo(1,2) + i4_ov(2,0) @ d_oo(1,3)
    ret["ovoo"] = i4_ov(3,1) @ d_oo(0,2) - i4_ov(2,1) @ d_oo(0,3)
    ret["oovo"] = - i5_ov(0,2) @ d_oo(1,3) + i5_ov(1,2) @ d_oo(0,3)
    ret["ooov"] = i5_ov(0,3) @ d_oo(1,2) - i5_ov(1,3) @ d_oo(0,2)
    """
    return ret




