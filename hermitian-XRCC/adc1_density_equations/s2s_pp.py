#import numpy as np
from qode.math.tensornet import evaluate, scalar_value, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, a, b, c, d = 'ijklabcd'        # lower the number of quotes we need to type

def s2s_2p_pp(d_oo, t2, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.ph.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.pphh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.ph.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.pphh.to_ndarray(), dtype=tl.float64))

    i1_vv = evaluate(ul1(i,0) @ ur1(i,1))
    i2_oo = evaluate(ul1(0,a) @ ur1(1,a))
    #i3_oovv = evaluate(scalar_value(ul1(i,a) @ ur1(i,a)) * t2(0,1,2,3))
    #i4_ov = evaluate(ur1(k,c) @ ul2(k,0,c,1))
    #i5_ov = evaluate(ul1(i,a) @ ur2(i,0,a,1))
 
    # zeroth order
    ret["vovo"] = ul1(3,0) @ ur1(1,2) - i1_vv(0,2) @ d_oo(1,3)
    ret["ovvo"] = - ul1(3,1) @ ur1(0,2) + i1_vv(1,2) @ d_oo(0,3)
    ret["voov"] = - ul1(2,0) @ ur1(1,3) + i1_vv(0,3) @ d_oo(1,2)
    ret["ovov"] = ul1(2,1) @ ur1(0,3) - i1_vv(1,3) @ d_oo(0,2)
    ret["oooo"] = (
        d_oo(1,3) @ i2_oo(2,0) - d_oo(0,3) @ i2_oo(2,1)
        - d_oo(1,2) @ i2_oo(3,0) + d_oo(0,2) @ i2_oo(3,1)
    )
 
    # first order
    ret["vvoo"] = (
        - i1_vv(0,b) @ t2(2,3,b,1) + i1_vv(1,b) @ t2(2,3,b,0)
        - i2_oo(2,j) @ t2(j,3,0,1) + i2_oo(3,j) @ t2(j,2,0,1)
        - ul1(2,1) @ ur1(j,b) @ t2(j,3,b,0)
        + ul1(3,1) @ ur1(j,b) @ t2(j,2,b,0)
        + ul1(2,0) @ ur1(j,b) @ t2(j,3,b,1)
        - ul1(3,0) @ ur1(j,b) @ t2(j,2,b,1)
        #+ i3_oovv(2,3,0,1)
    )
    ret["oovv"] = (
        - i1_vv(a,2) @ t2(0,1,a,3) + i1_vv(a,3) @ t2(0,1,a,2)
        - i2_oo(i,0) @ t2(i,1,2,3) + i2_oo(i,1) @ t2(i,0,2,3)
        - ur1(1,2) @ ul1(i,a) @ t2(i,0,a,3)
        + ur1(0,2) @ ul1(i,a) @ t2(i,1,a,3)
        + ur1(1,3) @ ul1(i,a) @ t2(i,0,a,2)
        - ur1(0,3) @ ul1(i,a) @ t2(i,1,a,2)
        #+ i3_oovv
    )
    """
    ret["vooo"] = 0.5 * ur1(1,c) @ ul2(2,3,c,0) + 0.5 * d_oo(2,1) @ i4_ov(3,0) - 0.5 * d_oo(3,1) @ i4_ov(2,0)
    ret["ovoo"] = - 0.5 * ur1(0,c) @ ul2(2,3,c,1) - d_oo(2,0) @ i4_ov(3,1) + d_oo(3,0) @ i4_ov(2,1)
    ret["vvov"] = 0.5 * ur1(k,3) @ ul2(k,2,0,1)
    ret["vvvo"] = - 0.5 * ur1(k,2) @ ul2(k,3,0,1)
    ret["vovv"] = 0.5 * ul1(i,0) @ ur2(1,i,2,3)
    ret["ovvv"] = - 0.5 * ul1(i,1) @ ur2(0,i,2,3)
    ret["ooov"] = 0.5 * ul1(2,a) @ ur2(0,1,a,3) - 0.5 * d_oo(2,0) @ i5_ov(1,3) + 0.5 * d_oo(2,1) @ i5_ov(0,3)
    ret["oovo"] = - 0.5 * ul1(3,a) @ ur2(0,1,a,2) + 0.5 * d_oo(3,0) @ i5_ov(1,2) - 0.5 * d_oo(3,1) @ i5_ov(0,2)    
    """
    return ret


