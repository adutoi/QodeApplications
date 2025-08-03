#import numpy as np
from qode.math.tensornet import evaluate, tl_tensor, scalar_value
from math import sqrt
import tensorly as tl

i, j, k, l, m, a, b, c, d, e = 'ijklmabcde'        # lower the number of quotes we need to type

def s2s_2p_2ip(t2, t2s, vec_left, vec_right):
    ret = {}
    ul1 = tl_tensor(tl.tensor(vec_left.h.to_ndarray(), dtype=tl.float64))
    ul2 = tl_tensor(tl.tensor(vec_left.phh.to_ndarray(), dtype=tl.float64))
    ur1 = tl_tensor(tl.tensor(vec_right.p.to_ndarray(), dtype=tl.float64))
    ur2 = tl_tensor(tl.tensor(vec_right.pph.to_ndarray(), dtype=tl.float64))
    t2_squared = scalar_value(t2(l,m,d,e) @ t2(l,m,d,e))

    # zeroth order
    ret["ov"] = (ul1(0) @ ur1(1)  # zeroth order
                 + 1 * t2(i,l,1,d) @ t2(0,l,a,d) @ ul1(i) @ ur1(a) - 0.5 * t2(i,l,d,e) @ t2(0,l,d,e) @ ul1(i) @ ur1(1) - 0.5 * t2(l,m,a,d) @ t2(l,m,1,d) @ ul1(0) @ ur1(a)  # generated
                 + 2 * ul2(0,j,a) @ ur2(j,a,1)
                 + 0.25 * t2_squared * ul1(0) @ ur1(1))
    ret["vo"] = (- ul1(1) @ ur1(0)  # zeroth order
                 + 0.5 * t2(i,l,d,e) @ t2(1,l,d,e) @ ul1(i) @ ur1(0) + 0.5 * t2(l,m,a,d) @ t2(l,m,0,d) @ ul1(1) @ ur1(a) - 1 * t2(i,l,0,d) @ t2(1,l,a,d) @ ul1(i) @ ur1(a)  # generated
                 - 2 * ul2(1,j,a) @ ur2(j,a,0)
                 - 0.25 * t2_squared * ul1(1) @ ur1(0))


    
    ret["oo"] = (sqrt(2) * ul2(0,1,a) @ ur1(a)  # first order
                 + t2s(0,a) @ ul1(1) @ ur1(a) - t2s(1,a) @ ul1(0) @ ur1(a)
                 + (1/sqrt(2)) * t2(0,1,a,b) @ ul1(i) @ ur2(i,a,b) + (1/sqrt(2)) * t2(j,0,a,b) @ ul1(1) @ ur2(j,a,b) - (1/sqrt(2)) * t2(j,1,a,b) @ ul1(0) @ ur2(j,a,b))
    ret["vv"] = (sqrt(2) * ul1(i) @ ur2(i,1,0)  # first order
                 + t2s(i,0) @ ul1(i) @ ur1(1) - t2s(i,1) @ ul1(i) @ ur1(0)
                 - (1/sqrt(2)) * t2(i,j,0,1) @ ul2(i,j,a) @ ur1(a) + (1/sqrt(2)) * t2(i,j,a,1) @ ul2(i,j,a) @ ur1(0) - (1/sqrt(2)) * t2(i,j,a,0) @ ul2(i,j,a) @ ur1(1))
    
    return ret


def s2s_3p_2ip(d_oo, t2, vec_left, vec_right):
    ret = {}
    ul1 = tl_tensor(tl.tensor(vec_left.h.to_ndarray(), dtype=tl.float64))
    #ul2 = tl_tensor(tl.tensor(vec_left.phh.to_ndarray(), dtype=tl.float64))
    ur1 = tl_tensor(tl.tensor(vec_right.p.to_ndarray(), dtype=tl.float64))
    #ur2 = tl_tensor(tl.tensor(vec_right.pph.to_ndarray(), dtype=tl.float64))
    #i1_oov = evaluate(ur1(a) @ t2(0,1,a,2))
    #i2_ovv = evaluate(ul1(i) @ t2(i,0,1,2))
    #i3_oo = evaluate(sqrt(2) * ur1(a) @ ul2(0,1,a))
    #i4_vv = evaluate(sqrt(2) * ul1(i) @ ur2(i,0,1))
    #ur1_sqrt2 = evaluate(sqrt(2) * ur1(0))
    #ul1_sqrt2 = evaluate(sqrt(2) * ul1(0))

    # zeroth order
    ret["ovoo"] = d_oo(0,2) @ ul1(3) @ ur1(1) - d_oo(0,3) @ ul1(2) @ ur1(1)
    ret["oovo"] = d_oo(0,3) @ ul1(1) @ ur1(2) - d_oo(0,1) @ ul1(3) @ ur1(2)
    ret["ooov"] = d_oo(0,1) @ ul1(2) @ ur1(3) - d_oo(0,2) @ ul1(1) @ ur1(3)

    # first order
    #ret["vooo"] = ul1(3) @ i1_oov(1,2,0) + ul1(1) @ i1_oov(2,3,0) + ul1(2) @ i1_oov(3,1,0)
    #ret["ovvv"] = ur1(3) @ i2_ovv(0,1,2) + ur1(1) @ i2_ovv(0,2,3) + ur1(2) @ i2_ovv(0,3,1)
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
