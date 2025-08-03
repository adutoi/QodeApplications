#import numpy as np
from qode.math.tensornet import evaluate, scalar_value, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, a, b, c, d = 'ijklabcd'        # lower the number of quotes we need to type

def s2s_1p_ip(t2, mp2_diffdm_ov, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.ph.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.pphh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.h.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.phh.to_ndarray(), dtype=tl.float64))
    #i1_ovov = evaluate(t2(0,l,1,d) @ t2(2,l,3,d))

    ret["v"] = (
        ul1(i) @ ur1(i,0)  # zeroth order
    )
    """
    + 0.25 * scalar_value(i1_ovov(l,d,l,d)) * ul1(i) @ ur1(i,0)
    - 0.5 * i1_ovov(i,d,j,d) @ ul1(i) @ ur1(j,0)
    + i1_ovov(i,0,j,a) @ ul1(i) @ ur1(j,a)
    - 0.5 * i1_ovov(l,a,l,0) @ ul1(i) @ ur1(i,a)
    + sqrt(2) * ul2(i,j,a) @ ur2(j,i,a,0)
    """
    #)

    """
    ret["o"] = (
        sqrt(2) * ul2(i,0,a) @ ur1(i,a)  # first order
        + ul1(0) @ ur1(j,a) @ mp2_diffdm_ov(j,a)
        - ul1(i) @ ur1(i,a) @ mp2_diffdm_ov(0,a)
    )
    """
    return ret


def s2s_2p_ip(d_oo, t2, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.ph.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.pphh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.h.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.phh.to_ndarray(), dtype=tl.float64))
    #i1_ovv = evaluate(sqrt(2) * ul2(i,0,1) @ ur1(i,2))
    #i2_o = evaluate(sqrt(2) * ul2(0,j,a) @ ur1(j,a))
    #i3_ooo = evaluate(np.sqrt(2) * ul2(0,1,a) @ ur1(2,a))

    # zeroth order
    ret["ovo"] = ul1(2) @ ur1(0,1) - d_oo(0,2) @ ul1(i) @ ur1(i,1)
    ret["oov"] = - ul1(1) @ ur1(0,2) + d_oo(0,1) @ ul1(i) @ ur1(i,2)

    # first order
    ret["voo"] = ul1(1) @ ur1(j,a) @ t2(j,2,a,0) - ul1(2) @ ur1(j,a) @ t2(j,1,a,0) - ul1(i) @ ur1(i,a) @ t2(1,2,a,0)
    """
    ret["vvo"] = i1_ovv(2,0,1)
    ret["vov"] = - i1_ovv(1,0,2)
    ret["ooo"] = i2_o(1) @ d_oo(0,2) - i2_o(2) @ d_oo(0,1) + sqrt(2) * ul2(0,1,a) @ ur1(2,a)  #+ i3_ooo(2,1,0)
    ret["ovv"] = 2 * ul1(i) @ ur2(i,0,1,2)
    """
    return ret


def s2s_3p_ip(d_oo, t2, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.ph.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.pphh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.h.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.phh.to_ndarray(), dtype=tl.float64))
    i1_v = evaluate(ul1(i) @ ur1(i,0))
    i2_vovv = evaluate(ur1(j,0) @ t2(j,1,2,3))
    i3_ooov = evaluate(ur1(0,a) @ t2(1,2,a,3))
    i4_oov = evaluate(ul1(i) @ ur1(i,a) @ t2(0,1,a,2))
    i5_ov = evaluate(ur1(j,a) @ t2(j,0,a,1))
    i6_ovv = evaluate(ul1(i) @ t2(i,0,1,2))
    #ul2_sqrt2 = evaluate(sqrt(2) * ul2(0,1,2))
    #i7_ooo = evaluate(ul2_sqrt2(0,1,a) @ ur1(2,a))
    #i8_ovv = evaluate(ul2_sqrt2(i,0,1) @ ur1(i,2))
    #i9_o = evaluate(ul2_sqrt2(i,0,a) @ ur1(i,a))
    #ur2_2 = evaluate(2 * ur2(0,1,2,3))
    #i10_ovv = evaluate(ul1(i) @ ur2_2(i,0,1,2))

    # zeroth order
    ret["oovoo"] = (
        ul1(3) @ ur1(0,2) @ d_oo(1,4)
        - ul1(3) @ ur1(1,2) @ d_oo(0,4)
        - ul1(4) @ ur1(0,2) @ d_oo(1,3)
        + ul1(4) @ ur1(1,2) @ d_oo(0,3)
        + i1_v(2) @ d_oo(1,3) @ d_oo(0,4)
        - i1_v(2) @ d_oo(0,3) @ d_oo(1,4)
    )
    ret["ooovo"] = (
        ul1(2) @ ur1(1,3) @ d_oo(0,4)
        - ul1(2) @ ur1(0,3) @ d_oo(1,4)
        - ul1(4) @ ur1(1,3) @ d_oo(0,2)
        + ul1(4) @ ur1(0,3) @ d_oo(1,2)
        + i1_v(3) @ d_oo(1,4) @ d_oo(0,2)
        - i1_v(3) @ d_oo(0,4) @ d_oo(1,2)
    )
    ret["oooov"] = (
        ul1(2) @ ur1(0,4) @ d_oo(1,3)
        - ul1(2) @ ur1(1,4) @ d_oo(0,3)
        - ul1(3) @ ur1(0,4) @ d_oo(1,2)
        + ul1(3) @ ur1(1,4) @ d_oo(0,2)
        + i1_v(4) @ d_oo(1,2) @ d_oo(0,3)
        - i1_v(4) @ d_oo(1,2) @ d_oo(0,3)
    )

    # first order
    ret["vvoov"] = i1_v(4) @ t2(2,3,0,1) + ul1(3) @ i2_vovv(4,2,0,1) - ul1(2) @ i2_vovv(4,3,0,1)
    ret["vvvoo"] = i1_v(2) @ t2(3,4,0,1) + ul1(4) @ i2_vovv(2,3,0,1) - ul1(3) @ i2_vovv(2,4,0,1)
    ret["vvovo"] = i1_v(3) @ t2(4,2,0,1) - ul1(4) @ i2_vovv(3,2,0,1) + ul1(2) @ i2_vovv(3,4,0,1)
    ret["voooo"] = (
        ul1(4) @ i3_ooov(1,2,3,0) - ul1(3) @ i3_ooov(1,2,4,0) + ul1(2) @ i3_ooov(1,3,4,0)
        - i4_oov(2,3,0) @ d_oo(1,4) + i4_oov(2,4,0) @ d_oo(1,3) - i4_oov(3,4,0) @ d_oo(1,2)
        + ul1(4) @ i5_ov(2,0) @ d_oo(1,3)
        - ul1(3) @ i5_ov(2,0) @ d_oo(1,4)
        + ul1(2) @ i5_ov(3,0) @ d_oo(1,4)
        - ul1(4) @ i5_ov(3,0) @ d_oo(1,2)
        + ul1(3) @ i5_ov(4,0) @ d_oo(1,2)
        - ul1(2) @ i5_ov(4,0) @ d_oo(1,3)
    )
    ret["ovooo"] = (
        - ul1(4) @ i3_ooov(0,2,3,1) + ul1(3) @ i3_ooov(0,2,4,1) - ul1(2) @ i3_ooov(0,3,4,1)
        + i4_oov(2,3,1) @ d_oo(0,4) - i4_oov(2,4,1) @ d_oo(0,3) + i4_oov(3,4,1) @ d_oo(0,2)
        - ul1(4) @ i5_ov(2,1) @ d_oo(0,3)
        + ul1(3) @ i5_ov(2,1) @ d_oo(0,4)
        - ul1(2) @ i5_ov(3,1) @ d_oo(0,4)
        + ul1(4) @ i5_ov(3,1) @ d_oo(0,2)
        - ul1(3) @ i5_ov(4,1) @ d_oo(0,2)
        + ul1(2) @ i5_ov(4,1) @ d_oo(0,3)
    )
    ret["oovvv"] = (
        i1_v(4) @ t2(0,1,2,3) - i1_v(3) @ t2(0,1,2,4) + i1_v(2) @ t2(0,1,3,4)
        + i6_ovv(0,2,3) @ ur1(1,4) - i6_ovv(0,2,4) @ ur1(1,3) + i6_ovv(0,3,4) @ ur1(1,2)
        - i6_ovv(1,2,3) @ ur1(0,4) + i6_ovv(1,2,4) @ ur1(0,3) - i6_ovv(1,3,4) @ ur1(0,2)
    )
    """
    ret["vooov"] = ul2_sqrt2(2,3,0) @ ur1(1,4) + i8_ovv(2,0,4) @ d_oo(1,3) - i8_ovv(3,0,4) @ d_oo(1,2)
    ret["voovo"] = - ul2_sqrt2(2,4,0) @ ur1(1,3) - i8_ovv(2,0,3) @ d_oo(1,4) + i8_ovv(4,0,3) @ d_oo(1,2)
    ret["vovoo"] = ul2_sqrt2(3,4,0) @ ur1(1,2) + i8_ovv(3,0,2) @ d_oo(1,4) - i8_ovv(4,0,2) @ d_oo(1,3)
    ret["ovoov"] = - ul2_sqrt2(2,3,1) @ ur1(0,4) - i8_ovv(2,1,4) @ d_oo(0,3) + i8_ovv(3,1,4) @ d_oo(0,2)
    ret["ovovo"] = ul2_sqrt2(2,4,1) @ ur1(0,3) + i8_ovv(2,1,3) @ d_oo(0,4) - i8_ovv(4,1,3) @ d_oo(0,2)
    ret["ovvoo"] = - ul2_sqrt2(3,4,1) @ ur1(0,2) - i8_ovv(3,1,2) @ d_oo(0,4) + i8_ovv(4,1,2) @ d_oo(0,3)
    ret["ooooo"] = (
        i7_ooo(2,3,1) @ d_oo(0,4) - i7_ooo(2,4,1) @ d_oo(0,3) + i7_ooo(3,4,1) @ d_oo(0,2)
        - i7_ooo(2,3,0) @ d_oo(1,4) + i7_ooo(2,4,0) @ d_oo(1,3) - i7_ooo(3,4,0) @ d_oo(1,2)
        + i9_o(2) @ d_oo(1,3) @ d_oo(0,4) - i9_o(3) @ d_oo(1,2) @ d_oo(0,4) + i9_o(4) @ d_oo(1,2) @ d_oo(0,3)
        - i9_o(2) @ d_oo(0,3) @ d_oo(1,4) + i9_o(3) @ d_oo(0,2) @ d_oo(1,4) - i9_o(4) @ d_oo(0,2) @ d_oo(1,3)
    )
    ret["oovvo"] = ul1(4) @ ur2_2(0,1,2,3) - i10_ovv(1,2,3) @ d_oo(0,4) + i10_ovv(0,2,3) @ d_oo(1,4)
    ret["oovov"] = - ul1(3) @ ur2_2(0,1,2,4) + i10_ovv(1,2,4) @ d_oo(0,3) - i10_ovv(0,2,4) @ d_oo(1,3)
    ret["ooovv"] = ul1(2) @ ur2_2(0,1,3,4) + i10_ovv(0,3,4) @ d_oo(1,2) - i10_ovv(1,3,4) @ d_oo(0,2)
    """
    return ret







