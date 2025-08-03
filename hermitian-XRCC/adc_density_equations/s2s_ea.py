#import numpy as np
from qode.math.tensornet import evaluate, scalar_value, tl_tensor
import tensorly as tl
from math import sqrt

i, j, k, l, m, a, b, c, d, e = 'ijklmabcde'        # lower the number of quotes we need to type

def s2s_1p_ea(t2, mp2_diffdm_ov, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.ph.to_ndarray(), dtype=tl.float64))
    ur2 = tl_tensor(tl.tensor(vec_right.pphh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.p.to_ndarray(), dtype=tl.float64))
    ul2 = tl_tensor(tl.tensor(vec_left.pph.to_ndarray(), dtype=tl.float64))
    i1_ovov = evaluate(t2(0,l,1,d) @ t2(2,l,3,d))

    ret["o"] = (
        ul1(a) @ ur1(0,a)  # zeroth order
        - 0.25 * scalar_value(i1_ovov(l,d,l,d)) * ul1(a) @ ur1(0,a)
        + 0.5 * i1_ovov(i,d,0,d) @ ul1(a) @ ur1(i,a)
        - i1_ovov(i,b,0,a) @ ul1(a) @ ur1(i,b)
        + 0.5 * i1_ovov(l,a,l,b) @ ul1(a) @ ur1(0,b)
        + sqrt(2) * ul2(i,a,b) @ ur2(0,i,a,b)    
    )

    
    ret["v"] = (
        sqrt(2) * ul2(i,a,0) @ ur1(i,a)  # first order
        + ul1(0) @ ur1(i,b) @ mp2_diffdm_ov(i,b)
        - ul1(a) @ ur1(i,a) @ mp2_diffdm_ov(i,0)
    )
    
    return ret


def s2s_2p_ea(d_oo, t2, t2s, t2d, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.ph.to_ndarray(), dtype=tl.float64))
    ur2 = tl_tensor(tl.tensor(vec_right.pphh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.p.to_ndarray(), dtype=tl.float64))
    ul2 = tl_tensor(tl.tensor(vec_left.pph.to_ndarray(), dtype=tl.float64))
    i1_ovo = evaluate(sqrt(2) * ul2(0,a,1) @ ur1(2,a))
    i2_v = evaluate(sqrt(2) * ul2(i,a,0) @ ur1(i,a))
    t2_squared = scalar_value(t2(l,m,d,e) @ t2(l,m,d,e))
    z = 1/sqrt(2)


    ret["vov"] = ul1(0) @ ur1(1,2)\
                  + 1 * t2(i,l,b,d) @ t2(1,l,2,d) @ ul1(0) @ ur1(i,b) + 1 * t2(i,l,0,d) @ t2(1,l,a,d) @ ul1(a) @ ur1(i,2) + 0.5 * t2(l,m,0,d) @ t2(l,m,2,d) @ ul1(a) @ ur1(1,a) + 1 * t2(i,l,b,0) @ t2(1,l,a,2) @ ul1(a) @ ur1(i,b) - 1 * t2(i,l,0,d) @ t2(1,l,2,d) @ ul1(a) @ ur1(i,a) - 0.5 * t2(i,l,d,e) @ t2(1,l,d,e) @ ul1(0) @ ur1(i,2) - 0.5 * t2(l,m,a,d) @ t2(l,m,0,d) @ ul1(a) @ ur1(1,2) - 0.5 * t2(l,m,b,d) @ t2(l,m,2,d) @ ul1(0) @ ur1(1,b) - 0.5 * t2(l,m,a,2) @ t2(l,m,b,0) @ ul1(a) @ ur1(1,b)\
                  + 0.25 * t2_squared * ul1(0) @ ur1(1,2)\
                  + z * ul2(i,a,0) @ ur2(i,1,a,2) + z * ul2(i,b,0) @ ur2(i,1,b,2) + 2 * z * ul2(i,0,d) @ ur2(i,1,2,d)  # generated
    ret["ovv"] = - ul1(1) @ ur1(0,2)\
                  + 1 * t2(i,l,1,d) @ t2(0,l,2,d) @ ul1(a) @ ur1(i,a) + 0.5 * t2(i,l,d,e) @ t2(0,l,d,e) @ ul1(1) @ ur1(i,2) + 0.5 * t2(l,m,a,d) @ t2(l,m,1,d) @ ul1(a) @ ur1(0,2) + 0.5 * t2(l,m,b,d) @ t2(l,m,2,d) @ ul1(1) @ ur1(0,b) + 0.5 * t2(l,m,a,2) @ t2(l,m,b,1) @ ul1(a) @ ur1(0,b) - 1 * t2(i,l,b,d) @ t2(0,l,2,d) @ ul1(1) @ ur1(i,b) - 1 * t2(i,l,1,d) @ t2(0,l,a,d) @ ul1(a) @ ur1(i,2) - 0.5 * t2(l,m,1,d) @ t2(l,m,2,d) @ ul1(a) @ ur1(0,a) - 1 * t2(i,l,b,1) @ t2(0,l,a,2) @ ul1(a) @ ur1(i,b)\
                  - 0.25 * t2_squared * ul1(1) @ ur1(0,2)\
                  - z * ul2(i,a,1) @ ur2(i,0,a,2) - z * ul2(i,b,1) @ ur2(i,0,b,2) - 2 * z * ul2(i,1,d) @ ur2(i,0,2,d)  # generated
    ret["ooo"] = d_oo(0,2) @ ul1(a) @ ur1(1,a) - d_oo(1,2) @ ul1(a) @ ur1(0,a)\
                  + 1 * t2(0,l,a,d) @ t2(2,l,b,d) @ ul1(a) @ ur1(1,b) + 0.5 * t2(1,l,d,e) @ t2(2,l,d,e) @ ul1(a) @ ur1(0,a) + 1 * t2(i,2,b,d) @ t2(0,1,a,d) @ ul1(a) @ ur1(i,b) - 0.5 * t2(0,l,d,e) @ t2(2,l,d,e) @ ul1(a) @ ur1(1,a) - 1 * t2(1,l,a,d) @ t2(2,l,b,d) @ ul1(a) @ ur1(0,b) - 0.5 * t2(i,2,d,e) @ t2(0,1,d,e) @ ul1(a) @ ur1(i,a) + 1 * d_oo(2,0) @ t2(i,l,b,d) @ t2(1,l,a,d) @ ul1(a) @ ur1(i,b) + 0.5 * d_oo(2,1) @ t2(i,l,d,e) @ t2(0,l,d,e) @ ul1(a) @ ur1(i,a) + 0.5 * d_oo(2,1) @ t2(l,m,a,d) @ t2(l,m,b,d) @ ul1(a) @ ur1(0,b) - 0.5 * d_oo(2,0) @ t2(i,l,d,e) @ t2(1,l,d,e) @ ul1(a) @ ur1(i,a) - 1 * d_oo(2,1) @ t2(i,l,b,d) @ t2(0,l,a,d) @ ul1(a) @ ur1(i,b) - 0.5 * d_oo(2,0) @ t2(l,m,a,d) @ t2(l,m,b,d) @ ul1(a) @ ur1(1,b)\
                  + 0.25 * t2_squared * ul1(a) @ ur1(1,a) @ d_oo(0,2) - 0.25 * t2_squared * ul1(a) @ ur1(0,a) @ d_oo(1,2)\
                  - z * ul2(2,a,d) @ ur2(0,1,a,d) - z * ul2(2,b,d) @ ur2(0,1,b,d) + z * d_oo(2,0) @ ul2(i,a,d) @ ur2(i,1,a,d) + z * d_oo(2,0) @ ul2(i,b,d) @ ur2(i,1,b,d) - z * d_oo(2,1) @ ul2(i,a,d) @ ur2(i,0,a,d) - z * d_oo(2,1) @ ul2(i,b,d) @ ur2(i,0,b,d)  # generated

    ret["vvo"] = ul1(1) @ ur1(i,b) @ t2(i,2,b,0) - ul1(0) @ ur1(i,b) @ t2(i,2,b,1) + ul1(a) @ ur1(i,a) @ t2(i,2,0,1)\
                  + ul1(1) @ ur1(i,b) @ t2d(i,2,b,0) - ul1(0) @ ur1(i,b) @ t2d(i,2,b,1) + ul1(a) @ ur1(i,a) @ t2d(i,2,0,1)
    
    ret["vvv"] = sqrt(2) * ul2(i,0,1) @ ur1(i,2)\
                  + 1 * ul1(0) @ ur1(i,2) @ t2s(i,1) - 1 * ul1(1) @ ur1(i,2) @ t2s(i,0)\
                  + 0.5 * ur2(i,j,b,2) @ t2(i,j,b,1) @ ul1(0) + 0.5 * ur2(i,j,c,2) @ t2(i,j,c,1) @ ul1(0) - 0.5 * ur2(i,j,b,2) @ t2(i,j,b,0) @ ul1(1) - 0.5 * ur2(i,j,c,2) @ t2(i,j,c,0) @ ul1(1) - 1 * ur2(i,j,a,2) @ t2(i,j,0,1) @ ul1(a)  # generated
    ret["ovo"] = i2_v(1) @ d_oo(0,2) - i1_ovo(2,1,0)\
                  + 1 * ul1(1) @ ur1(0,b) @ t2s(2,b) - 1 * ul1(a) @ ur1(0,a) @ t2s(2,1) + 1 * d_oo(2,0) @ ul1(a) @ ur1(i,a) @ t2s(i,1) - 1 * d_oo(2,0) @ ul1(1) @ ur1(i,b) @ t2s(i,b)\
                  + 0.5 * ur2(i,0,a,b) @ t2(i,2,b,1) @ ul1(a) + 0.5 * ur2(i,0,a,c) @ t2(i,2,c,1) @ ul1(a) + 0.5 * ur2(i,0,b,c) @ t2(i,2,b,c) @ ul1(1) + 0.5 * ur2(j,0,a,b) @ t2(j,2,b,1) @ ul1(a) + 0.5 * ur2(j,0,a,c) @ t2(j,2,c,1) @ ul1(a) + 0.5 * ur2(j,0,b,c) @ t2(j,2,b,c) @ ul1(1) - 0.5 * d_oo(2,0) @ ur2(i,j,a,b) @ t2(i,j,b,1) @ ul1(a) - 0.5 * d_oo(2,0) @ ur2(i,j,a,c) @ t2(i,j,c,1) @ ul1(a) - 0.5 * d_oo(2,0) @ ur2(i,j,b,c) @ t2(i,j,b,c) @ ul1(1)  # generated
    ret["voo"] = - i2_v(0) @ d_oo(1,2) + i1_ovo(2,0,1)\
                  + 1 * ul1(a) @ ur1(1,a) @ t2s(2,0) - 1 * ul1(0) @ ur1(1,b) @ t2s(2,b) + 1 * d_oo(2,1) @ ul1(0) @ ur1(i,b) @ t2s(i,b) - 1 * d_oo(2,1) @ ul1(a) @ ur1(i,a) @ t2s(i,0)\
                  - 0.5 * ur2(i,1,a,b) @ t2(i,2,b,0) @ ul1(a) - 0.5 * ur2(i,1,a,c) @ t2(i,2,c,0) @ ul1(a) - 0.5 * ur2(i,1,b,c) @ t2(i,2,b,c) @ ul1(0) - 0.5 * ur2(j,1,a,b) @ t2(j,2,b,0) @ ul1(a) - 0.5 * ur2(j,1,a,c) @ t2(j,2,c,0) @ ul1(a) - 0.5 * ur2(j,1,b,c) @ t2(j,2,b,c) @ ul1(0) + 0.5 * d_oo(2,1) @ ur2(i,j,a,b) @ t2(i,j,b,0) @ ul1(a) + 0.5 * d_oo(2,1) @ ur2(i,j,a,c) @ t2(i,j,c,0) @ ul1(a) + 0.5 * d_oo(2,1) @ ur2(i,j,b,c) @ t2(i,j,b,c) @ ul1(0)  # generated
    ret["oov"] = 2 * ul1(a) @ ur2(1,0,2,a)\
                  + 1 * ul1(a) @ ur1(0,2) @ t2s(1,a) + 1 * ul1(a) @ ur1(1,a) @ t2s(0,2) - 1 * ul1(a) @ ur1(0,a) @ t2s(1,2) - 1 * ul1(a) @ ur1(1,2) @ t2s(0,a)\
                  + z * ul2(i,a,b) @ t2(0,1,a,2) @ ur1(i,b) + z * ul2(i,a,b) @ t2(i,0,a,2) @ ur1(1,b) + z * ul2(i,a,b) @ t2(i,1,a,b) @ ur1(0,2) + z * ul2(i,a,b) @ t2(i,1,b,2) @ ur1(0,a) - z * ul2(i,a,b) @ t2(0,1,a,b) @ ur1(i,2) - z * ul2(i,a,b) @ t2(0,1,b,2) @ ur1(i,a) - z * ul2(i,a,b) @ t2(i,0,a,b) @ ur1(1,2) - z * ul2(i,a,b) @ t2(i,0,b,2) @ ur1(1,a) - z * ul2(i,a,b) @ t2(i,1,a,2) @ ur1(0,b)  # generated
    
    return ret


def s2s_3p_ea(d_oo, t2, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.ph.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.pphh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.p.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.pph.to_ndarray(), dtype=tl.float64))
    i1_o = evaluate(ul1(a) @ ur1(0,a))
    i2_vovv = evaluate(ur1(j,0) @ t2(j,1,2,3))
    i3_ooov = evaluate(ur1(0,a) @ t2(1,2,a,3))
    i4_ovv = evaluate(ul1(a) @ ur1(i,a) @ t2(i,0,1,2))
    i5_ov = evaluate(ur1(j,a) @ t2(j,0,a,1))
    i6_oov = evaluate(ul1(a) @ t2(0,1,a,2))
    #ul2_sqrt2 = evaluate(sqrt(2) * ul2(0,1,2))
    #i7_vvv = evaluate(ul2_sqrt2(i,0,1) @ ur1(i,2))
    #i8_ovo = evaluate(ul2_sqrt2(0,a,1) @ ur1(2,a))
    #i9_v = evaluate(ul2_sqrt2(i,a,0) @ ur1(i,a))
    #ur2_2 = evaluate(2 * ur2(0,1,2,3))
    #i10_oov = evaluate(2 * ul1(a) @ ur2_2(0,1,a,2))

    # zeroth order
    ret["vooov"] = (
        ul1(0) @ ur1(1,4) @ d_oo(2,3)
        - ul1(0) @ ur1(2,4) @ d_oo(1,3)
    )
    ret["ovoov"] = (
        - ul1(1) @ ur1(0,4) @ d_oo(2,3)
        + ul1(1) @ ur1(2,4) @ d_oo(0,3)
    )
    ret["oovov"] = (
        ul1(2) @ ur1(0,4) @ d_oo(1,3)
        - ul1(2) @ ur1(1,4) @ d_oo(0,3)
    )
    ret["voovo"] = (
        - ul1(0) @ ur1(1,3) @ d_oo(2,4)
        + ul1(0) @ ur1(2,3) @ d_oo(1,4)
    )
    ret["ovovo"] = (
        ul1(1) @ ur1(0,3) @ d_oo(2,4)
        - ul1(1) @ ur1(2,3) @ d_oo(0,4)
    )
    ret["oovvo"] = (
        - ul1(2) @ ur1(0,3) @ d_oo(1,4)
        + ul1(2) @ ur1(1,3) @ d_oo(0,4)
    )
    ret["ooooo"] = (
        i1_o(0) @ d_oo(1,3) @ d_oo(2,4)
        + i1_o(1) @ d_oo(2,3) @ d_oo(0,4)
        + i1_o(2) @ d_oo(1,4) @ d_oo(0,3)
        - i1_o(0) @ d_oo(1,4) @ d_oo(2,3)
        - i1_o(1) @ d_oo(2,4) @ d_oo(0,3)
        - i1_o(2) @ d_oo(1,3) @ d_oo(0,4)
    )

    # first order
    
    ret["vovoo"] = (
        i1_o(1) @ t2(3,4,0,2)
        + ul1(2) @ i3_ooov(1,3,4,0) - ul1(0) @ i3_ooov(1,3,4,2)
        + i4_ovv(3,0,2) @ d_oo(1,4) - i4_ovv(4,0,2) @ d_oo(1,3)
        + ul1(2) @ i5_ov(3,0) @ d_oo(1,4) - ul1(2) @ i5_ov(4,0) @ d_oo(1,3)
        - ul1(0) @ i5_ov(3,2) @ d_oo(1,4) + ul1(0) @ i5_ov(4,2) @ d_oo(1,3)
    )
    ret["vvooo"] = (
        - i1_o(2) @ t2(3,4,0,1)
        - ul1(1) @ i3_ooov(2,3,4,0) + ul1(0) @ i3_ooov(2,3,4,1)
        - i4_ovv(3,0,1) @ d_oo(2,4) + i4_ovv(4,0,1) @ d_oo(2,3)
        - ul1(1) @ i5_ov(3,0) @ d_oo(2,4) + ul1(1) @ i5_ov(4,0) @ d_oo(2,3)
        + ul1(0) @ i5_ov(3,1) @ d_oo(2,4) - ul1(0) @ i5_ov(4,1) @ d_oo(2,3)
    )
    ret["ovvoo"] = (
        - i1_o(0) @ t2(3,4,1,2)
        - ul1(2) @ i3_ooov(0,3,4,1) + ul1(1) @ i3_ooov(0,3,4,2)
        + i4_ovv(4,1,2) @ d_oo(0,3) - i4_ovv(3,1,2) @ d_oo(0,4)
        - ul1(2) @ i5_ov(3,1) @ d_oo(0,4) + ul1(2) @ i5_ov(4,1) @ d_oo(0,3)
        + ul1(1) @ i5_ov(3,2) @ d_oo(0,4) - ul1(1) @ i5_ov(4,2) @ d_oo(0,3)
    )
    ret["vvvov"] = (
        ul1(1) @ i2_vovv(4,3,0,2) - ul1(2) @ i2_vovv(4,3,0,1) - ul1(0) @ i2_vovv(4,3,1,2)
    )
    ret["vvvvo"] = (
        - ul1(1) @ i2_vovv(3,4,0,2) + ul1(2) @ i2_vovv(3,4,0,1) + ul1(0) @ i2_vovv(3,4,1,2)
    )
    ret["ooovv"] = (
        i1_o(1) @ t2(0,2,3,4) - i1_o(2) @ t2(0,1,3,4) + i1_o(0) @ t2(1,2,3,4)
        + ur1(2,3) @ i6_oov(0,1,4) - ur1(1,3) @ i6_oov(0,2,4) + ur1(0,3) @ i6_oov(1,2,4)
        - ur1(2,4) @ i6_oov(0,1,3) + ur1(1,4) @ i6_oov(0,2,3) - ur1(0,4) @ i6_oov(1,2,3)
    )
    
    """
    ret["vovov"] = ul2_sqrt2(3,0,2) @ ur1(1,4) - i7_vvv(0,2,4) @ d_oo(1,3)
    ret["vvoov"] = - ul2_sqrt2(3,0,1) @ ur1(2,4) + i7_vvv(0,1,4) @ d_oo(2,3)
    ret["ovvov"] = - ul2_sqrt2(3,1,2) @ ur1(0,4) + i7_vvv(1,2,4) @ d_oo(0,3)
    ret["vovvo"] = - ul2_sqrt2(4,0,2) @ ur1(1,3) + i7_vvv(0,2,3) @ d_oo(1,4)
    ret["vvovo"] = ul2_sqrt2(4,0,1) @ ur1(2,3) - i7_vvv(0,1,3) @ d_oo(2,4)
    ret["ovvvo"] = ul2_sqrt2(4,1,2) @ ur1(0,3) - i7_vvv(1,2,3) @ d_oo(0,4)
    ret["voooo"] = (
        i8_ovo(3,0,2) @ d_oo(1,4)
        - i8_ovo(3,0,1) @ d_oo(2,4)
        - i8_ovo(4,0,2) @ d_oo(1,3)
        + i8_ovo(4,0,1) @ d_oo(2,3)
        + i9_v(0) @ d_oo(1,3) @ d_oo(2,4)
        - i9_v(0) @ d_oo(1,4) @ d_oo(2,3)
    )
    ret["ovooo"] = (
        i8_ovo(3,1,0) @ d_oo(2,4)
        - i8_ovo(3,1,2) @ d_oo(0,4)
        - i8_ovo(4,1,0) @ d_oo(2,3)
        + i8_ovo(4,1,2) @ d_oo(0,3)
        - i9_v(1) @ d_oo(0,3) @ d_oo(2,4)
        + i9_v(1) @ d_oo(0,4) @ d_oo(2,3)
    )
    ret["oovoo"] = (
        i8_ovo(3,2,1) @ d_oo(0,4)
        - i8_ovo(3,2,0) @ d_oo(1,4)
        - i8_ovo(4,2,1) @ d_oo(0,3)
        + i8_ovo(4,2,0) @ d_oo(1,3)
        + i9_v(2) @ d_oo(0,3) @ d_oo(1,4)
        - i9_v(2) @ d_oo(0,4) @ d_oo(1,3)
    )
    ret["oovvv"] = ul1(2) @ ur2_2(0,1,3,4)
    ret["ovovv"] = - ul1(1) @ ur2_2(0,2,3,4)
    ret["voovv"] = ul1(0) @ ur2_2(1,2,3,4)
    ret["ooovo"] = i10_oov(0,2,3) @ d_oo(1,4) - i10_oov(0,1,3) @ d_oo(2,4) - i10_oov(1,2,3) @ d_oo(0,4)
    ret["oooov"] = i10_oov(0,2,4) @ d_oo(1,3) - i10_oov(0,1,4) @ d_oo(2,3) - i10_oov(1,2,4) @ d_oo(0,3)
    """
    return ret







