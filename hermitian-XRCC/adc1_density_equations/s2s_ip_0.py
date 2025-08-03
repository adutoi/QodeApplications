#import numpy as np
from qode.math.tensornet import evaluate, scalar_value, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, a, b, c, d = 'ijklabcd'        # lower the number of quotes we need to type

def s2s_2p_ip_0(d_oo, t2, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.h.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.phh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.h.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.phh.to_ndarray(), dtype=tl.float64))
    #i1 = scalar_value(ul1(i) @ ur1(i))
    i2_ovv_r = evaluate(ur1(j) @ t2(j,0,1,2))
    i3_ovv_l = evaluate(ul1(i) @ t2(i,0,1,2))
    #ur1_sqrt2 = evaluate(sqrt(2) * ur1)
    #ul1_sqrt2 = evaluate(sqrt(2) * ul1)
    #i4_ov = evaluate(ul2(0,j,1) @ ur1_sqrt2(j))
    #i5_ov = evaluate(ul1_sqrt2(i) @ ur2(i,0,1))

    # zeroth order
    ret["oooo"] = (
        ur1(0) @ ul1(2) @ d_oo(1,3)
        - ur1(1) @ ul1(2) @ d_oo(0,3)
        - ur1(0) @ ul1(3) @ d_oo(1,2)
        + ur1(1) @ ul1(3) @ d_oo(0,2)
        #+ i1 * d_oo(0,3) @ d_oo(1,2)
        #- i1 * d_oo(1,3) @ d_oo(0,2)
    )

    # first order
    ret["vvoo"] = ul1(3) @ i2_ovv_r(2,0,1) - ul1(2) @ i2_ovv_r(3,0,1)
    ret["oovv"] = ur1(1) @ i3_ovv_l(0,2,3) - ur1(0) @ i3_ovv_l(1,2,3)
    """
    ret["vooo"] = ul2(2,3,0) @ ur1_sqrt2(1) + i4_ov(3,0) @ d_oo(1,2) - i4_ov(2,0) @ d_oo(1,3)
    ret["ovoo"] = - ul2(2,3,1) @ ur1_sqrt2(0) - i4_ov(3,1) @ d_oo(0,2) + i4_ov(2,1) @ d_oo(0,3)
    ret["oovo"] = ul1_sqrt2(3) @ ur2(0,1,2) + i5_ov(0,2) @ d_oo(1,3) - i5_ov(1,2) @ d_oo(0,3)
    ret["ooov"] = - ul1_sqrt2(2) @ ur2(0,1,3) - i5_ov(0,3) @ d_oo(1,2) + i5_ov(1,3) @ d_oo(0,2)
    """
    return ret




