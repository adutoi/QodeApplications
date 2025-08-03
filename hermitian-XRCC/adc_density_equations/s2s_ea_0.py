#import numpy as np
from qode.math.tensornet import evaluate, scalar_value, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, m, a, b, c, d, e = 'ijklmabcde'        # lower the number of quotes we need to type

def s2s_2p_ea_0(d_oo, t2, t2s, t2d, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.p.to_ndarray(), dtype=tl.float64))
    ur2 = tl_tensor(tl.tensor(vec_right.pph.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.p.to_ndarray(), dtype=tl.float64))
    ul2 = tl_tensor(tl.tensor(vec_left.pph.to_ndarray(), dtype=tl.float64))
    i2_oov_r = evaluate(ur1(b) @ t2(0,1,b,2))
    i3_oov_l = evaluate(ul1(a) @ t2(0,1,a,2))
    i2_oov_rd = evaluate(ur1(b) @ t2d(0,1,b,2))
    i3_oov_ld = evaluate(ul1(a) @ t2d(0,1,a,2))
    ur1_sqrt2 = evaluate(sqrt(2) * ur1)
    ul1_sqrt2 = evaluate(sqrt(2) * ul1)
    i4_ov = evaluate(ul2(0,1,b) @ ur1_sqrt2(b))
    i5_ov = evaluate(ul1_sqrt2(a) @ ur2(0,a,1))
    t2_squared = scalar_value(t2(l,m,d,e) @ t2(l,m,d,e))


    ret["oooo"] = (
        + 1 * t2(0,1,a,d) @ t2(2,3,b,d) @ ul1(a) @ ur1(b) + 1 * d_oo(3,0) @ t2(1,l,a,d) @ t2(2,l,b,d) @ ul1(a) @ ur1(b) + 1 * d_oo(2,1) @ t2(0,l,a,d) @ t2(3,l,b,d) @ ul1(a) @ ur1(b) - 1 * d_oo(2,0) @ t2(1,l,a,d) @ t2(3,l,b,d) @ ul1(a) @ ur1(b) - 1 * d_oo(3,1) @ t2(0,l,a,d) @ t2(2,l,b,d) @ ul1(a) @ ur1(b)  # generated
        + 0.5 * d_oo(3,1) @ ul2(2,a,d) @ ur2(0,a,d) + 0.5 * d_oo(3,1) @ ul2(2,b,d) @ ur2(0,b,d) + 0.5 * d_oo(2,0) @ ul2(3,a,d) @ ur2(1,a,d) + 0.5 * d_oo(2,0) @ ul2(3,b,d) @ ur2(1,b,d) - 0.5 * d_oo(3,0) @ ul2(2,a,d) @ ur2(1,a,d) - 0.5 * d_oo(3,0) @ ul2(2,b,d) @ ur2(1,b,d) - 0.5 * d_oo(2,1) @ ul2(3,a,d) @ ur2(0,a,d) - 0.5 * d_oo(2,1) @ ul2(3,b,d) @ ur2(0,b,d)  # generated
    )
    ret["voov"] = (ul1(0) @ ur1(3) @ d_oo(1,2)
                   + 1 * t2(1,l,3,d) @ t2(2,l,b,d) @ ul1(0) @ ur1(b) + 1 * t2(1,l,a,d) @ t2(2,l,0,d) @ ul1(a) @ ur1(3) + 1 * t2(1,l,a,3) @ t2(2,l,b,0) @ ul1(a) @ ur1(b) - 0.5 * t2(1,l,d,e) @ t2(2,l,d,e) @ ul1(0) @ ur1(3) - 0.5 * d_oo(2,1) @ t2(l,m,b,d) @ t2(l,m,3,d) @ ul1(0) @ ur1(b) - 0.5 * d_oo(2,1) @ t2(l,m,a,d) @ t2(l,m,0,d) @ ul1(a) @ ur1(3) - 0.5 * d_oo(2,1) @ t2(l,m,a,3) @ t2(l,m,b,0) @ ul1(a) @ ur1(b)  # generated
                   + 0.25 * t2_squared * ul1(0) @ ur1(3) @ d_oo(1,2)
                   - 0.5 * ul2(2,a,0) @ ur2(1,a,3) - 0.5 * ul2(2,b,0) @ ur2(1,b,3) - 1 * ul2(2,0,d) @ ur2(1,3,d) + 0.5 * d_oo(2,1) @ ul2(i,a,0) @ ur2(i,a,3) + 0.5 * d_oo(2,1) @ ul2(i,b,0) @ ur2(i,b,3) + 1 * d_oo(2,1) @ ul2(i,0,d) @ ur2(i,3,d))  # generated
    ret["ovov"] = (- ul1(1) @ ur1(3) @ d_oo(0,2)  # zeroth order
                   + 0.5 * t2(0,l,d,e) @ t2(2,l,d,e) @ ul1(1) @ ur1(3) - 1 * t2(0,l,3,d) @ t2(2,l,b,d) @ ul1(1) @ ur1(b) - 1 * t2(0,l,a,d) @ t2(2,l,1,d) @ ul1(a) @ ur1(3) - 1 * t2(0,l,a,3) @ t2(2,l,b,1) @ ul1(a) @ ur1(b) + 0.5 * d_oo(2,0) @ t2(l,m,b,d) @ t2(l,m,3,d) @ ul1(1) @ ur1(b) + 0.5 * d_oo(2,0) @ t2(l,m,a,d) @ t2(l,m,1,d) @ ul1(a) @ ur1(3) + 0.5 * d_oo(2,0) @ t2(l,m,a,3) @ t2(l,m,b,1) @ ul1(a) @ ur1(b)  # generated
                   - 0.25 * t2_squared * ul1(1) @ ur1(3) @ d_oo(0,2)
                   + 0.5 * ul2(2,a,1) @ ur2(0,a,3) + 0.5 * ul2(2,b,1) @ ur2(0,b,3) + 1 * ul2(2,1,d) @ ur2(0,3,d) - 0.5 * d_oo(2,0) @ ul2(i,a,1) @ ur2(i,a,3) - 0.5 * d_oo(2,0) @ ul2(i,b,1) @ ur2(i,b,3) - 1 * d_oo(2,0) @ ul2(i,1,d) @ ur2(i,3,d))  # generated
    ret["vovo"] = (- ul1(0) @ ur1(2) @ d_oo(1,3)
                   + 0.5 * t2(1,l,d,e) @ t2(3,l,d,e) @ ul1(0) @ ur1(2) - 1 * t2(1,l,2,d) @ t2(3,l,b,d) @ ul1(0) @ ur1(b) - 1 * t2(1,l,a,d) @ t2(3,l,0,d) @ ul1(a) @ ur1(2) - 1 * t2(1,l,a,2) @ t2(3,l,b,0) @ ul1(a) @ ur1(b) + 0.5 * d_oo(3,1) @ t2(l,m,b,d) @ t2(l,m,2,d) @ ul1(0) @ ur1(b) + 0.5 * d_oo(3,1) @ t2(l,m,a,d) @ t2(l,m,0,d) @ ul1(a) @ ur1(2) + 0.5 * d_oo(3,1) @ t2(l,m,a,2) @ t2(l,m,b,0) @ ul1(a) @ ur1(b)  # generated
                   - 0.25 * t2_squared * ul1(0) @ ur1(2) @ d_oo(1,3)
                   + 0.5 * ul2(3,a,0) @ ur2(1,a,2) + 0.5 * ul2(3,b,0) @ ur2(1,b,2) + 1 * ul2(3,0,d) @ ur2(1,2,d) - 0.5 * d_oo(3,1) @ ul2(i,a,0) @ ur2(i,a,2) - 0.5 * d_oo(3,1) @ ul2(i,b,0) @ ur2(i,b,2) - 1 * d_oo(3,1) @ ul2(i,0,d) @ ur2(i,2,d))  # generated
    ret["ovvo"] = (ul1(1) @ ur1(2) @ d_oo(0,3)  # zeroth order
                   + 1 * t2(0,l,2,d) @ t2(3,l,b,d) @ ul1(1) @ ur1(b) + 1 * t2(0,l,a,d) @ t2(3,l,1,d) @ ul1(a) @ ur1(2) + 1 * t2(0,l,a,2) @ t2(3,l,b,1) @ ul1(a) @ ur1(b) - 0.5 * t2(0,l,d,e) @ t2(3,l,d,e) @ ul1(1) @ ur1(2) - 0.5 * d_oo(3,0) @ t2(l,m,b,d) @ t2(l,m,2,d) @ ul1(1) @ ur1(b) - 0.5 * d_oo(3,0) @ t2(l,m,a,d) @ t2(l,m,1,d) @ ul1(a) @ ur1(2) - 0.5 * d_oo(3,0) @ t2(l,m,a,2) @ t2(l,m,b,1) @ ul1(a) @ ur1(b)  # generated
                   + 0.25 * t2_squared * ul1(1) @ ur1(2) @ d_oo(0,3)
                   - 0.5 * ul2(3,a,1) @ ur2(0,a,2) - 0.5 * ul2(3,b,1) @ ur2(0,b,2) - 1 * ul2(3,1,d) @ ur2(0,2,d) + 0.5 * d_oo(3,0) @ ul2(i,a,1) @ ur2(i,a,2) + 0.5 * d_oo(3,0) @ ul2(i,b,1) @ ur2(i,b,2) + 1 * d_oo(3,0) @ ul2(i,1,d) @ ur2(i,2,d))  # generated

    ret["vvoo"] = (ul1(1) @ i2_oov_r(2,3,0) - ul1(0) @ i2_oov_r(2,3,1)  # first order
                   + ul1(1) @ i2_oov_rd(2,3,0) - ul1(0) @ i2_oov_rd(2,3,1))
    ret["oovv"] = (ur1(3) @ i3_oov_l(0,1,2) - ur1(2) @ i3_oov_l(0,1,3)  # first order
                   + ur1(3) @ i3_oov_ld(0,1,2) - ur1(2) @ i3_oov_ld(0,1,3))
    
    ret["vvov"] = (ul2(2,0,1) @ ur1_sqrt2(3)
                   + 1 * ul1(0) @ ur1(3) @ t2s(2,1) - 1 * ul1(1) @ ur1(3) @ t2s(2,0)  # generated
                   + (1/sqrt(2)) * (ur2(i,b,3) @ t2(i,2,b,1) @ ul1(0) + 1 * ur2(i,c,3) @ t2(i,2,c,1) @ ul1(0) - 1 * ur2(i,b,3) @ t2(i,2,b,0) @ ul1(1) - 1 * ur2(i,c,3) @ t2(i,2,c,0) @ ul1(1) - 2 * ur2(i,a,3) @ t2(i,2,0,1) @ ul1(a)))  # generated
    ret["vvvo"] = (- ul2(3,0,1) @ ur1_sqrt2(2)
                   + 1 * ul1(1) @ ur1(2) @ t2s(3,0) - 1 * ul1(0) @ ur1(2) @ t2s(3,1)  # generated
                   + (1/sqrt(2)) * (ur2(i,b,2) @ t2(i,3,b,0) @ ul1(1) + 1 * ur2(i,c,2) @ t2(i,3,c,0) @ ul1(1) + 2 * ur2(i,a,2) @ t2(i,3,0,1) @ ul1(a) - 1 * ur2(i,b,2) @ t2(i,3,b,1) @ ul1(0) - 1 * ur2(i,c,2) @ t2(i,3,c,1) @ ul1(0)))  # generated
    ret["vovv"] = (ul1_sqrt2(0) @ ur2(1,3,2)
                   + 1 * ul1(0) @ ur1(3) @ t2s(1,2) - 1 * ul1(0) @ ur1(2) @ t2s(1,3)  # generated
                   + (1/sqrt(2)) * (ul2(i,a,0) @ t2(i,1,2,3) @ ur1(a) + 1 * ul2(i,a,0) @ t2(i,1,a,2) @ ur1(3) + 1 * ul2(i,b,0) @ t2(i,1,2,3) @ ur1(b) + 1 * ul2(i,b,0) @ t2(i,1,b,2) @ ur1(3) - 1 * ul2(i,a,0) @ t2(i,1,a,3) @ ur1(2) - 1 * ul2(i,b,0) @ t2(i,1,b,3) @ ur1(2)))  # generated
    ret["ovvv"] = (- ul1_sqrt2(1) @ ur2(0,3,2)
                   + 1 * ul1(1) @ ur1(2) @ t2s(0,3) - 1 * ul1(1) @ ur1(3) @ t2s(0,2)  # generated
                   + (1/sqrt(2)) * (ul2(i,a,1) @ t2(i,0,a,3) @ ur1(2) + 1 * ul2(i,b,1) @ t2(i,0,b,3) @ ur1(2) - 1 * ul2(i,a,1) @ t2(i,0,2,3) @ ur1(a) - 1 * ul2(i,a,1) @ t2(i,0,a,2) @ ur1(3) - 1 * ul2(i,b,1) @ t2(i,0,2,3) @ ur1(b) - 1 * ul2(i,b,1) @ t2(i,0,b,2) @ ur1(3)))  # generated
    ret["vooo"] = (- i4_ov(3,0) @ d_oo(1,2) + i4_ov(2,0) @ d_oo(1,3)
                   + 1 * d_oo(3,1) @ ul1(0) @ ur1(b) @ t2s(2,b) - 1 * d_oo(2,1) @ ul1(0) @ ur1(b) @ t2s(3,b)  # generated
                   + (1/sqrt(2)) * (ur2(1,a,b) @ t2(2,3,b,0) @ ul1(a) + 1 * ur2(1,a,c) @ t2(2,3,c,0) @ ul1(a) + 1 * ur2(1,b,c) @ t2(2,3,b,c) @ ul1(0) + 1 * d_oo(3,1) @ ur2(i,a,b) @ t2(i,2,b,0) @ ul1(a) + 1 * d_oo(3,1) @ ur2(i,a,c) @ t2(i,2,c,0) @ ul1(a) + 1 * d_oo(3,1) @ ur2(i,b,c) @ t2(i,2,b,c) @ ul1(0) - 1 * d_oo(2,1) @ ur2(i,a,b) @ t2(i,3,b,0) @ ul1(a) - 1 * d_oo(2,1) @ ur2(i,a,c) @ t2(i,3,c,0) @ ul1(a) - 1 * d_oo(2,1) @ ur2(i,b,c) @ t2(i,3,b,c) @ ul1(0)))  # generated
    ret["ovoo"] = (i4_ov(3,1) @ d_oo(0,2) - i4_ov(2,1) @ d_oo(0,3)
                   + 1 * d_oo(2,0) @ ul1(1) @ ur1(b) @ t2s(3,b) - 1 * d_oo(3,0) @ ul1(1) @ ur1(b) @ t2s(2,b)  # generated
                   + (1/sqrt(2)) * (- 1 * ur2(0,a,b) @ t2(2,3,b,1) @ ul1(a) - 1 * ur2(0,a,c) @ t2(2,3,c,1) @ ul1(a) - 1 * ur2(0,b,c) @ t2(2,3,b,c) @ ul1(1) + 1 * d_oo(2,0) @ ur2(i,a,b) @ t2(i,3,b,1) @ ul1(a) + 1 * d_oo(2,0) @ ur2(i,a,c) @ t2(i,3,c,1) @ ul1(a) + 1 * d_oo(2,0) @ ur2(i,b,c) @ t2(i,3,b,c) @ ul1(1) - 1 * d_oo(3,0) @ ur2(i,a,b) @ t2(i,2,b,1) @ ul1(a) - 1 * d_oo(3,0) @ ur2(i,a,c) @ t2(i,2,c,1) @ ul1(a) - 1 * d_oo(3,0) @ ur2(i,b,c) @ t2(i,2,b,c) @ ul1(1)))  # generated
    ret["oovo"] = (- i5_ov(0,2) @ d_oo(1,3) + i5_ov(1,2) @ d_oo(0,3)
                   + 1 * d_oo(3,1) @ ul1(a) @ ur1(2) @ t2s(0,a) - 1 * d_oo(3,0) @ ul1(a) @ ur1(2) @ t2s(1,a)  # generated
                   + (1/sqrt(2)) * (ul2(3,a,b) @ t2(0,1,a,b) @ ur1(2) + 1 * ul2(3,a,b) @ t2(0,1,b,2) @ ur1(a) - 1 * ul2(3,a,b) @ t2(0,1,a,2) @ ur1(b) + 1 * d_oo(3,0) @ ul2(i,a,b) @ t2(i,1,a,2) @ ur1(b) + 1 * d_oo(3,1) @ ul2(i,a,b) @ t2(i,0,b,2) @ ur1(a) + 1 * d_oo(3,1) @ ul2(i,a,b) @ t2(i,0,a,b) @ ur1(2) - 1 * d_oo(3,0) @ ul2(i,a,b) @ t2(i,1,b,2) @ ur1(a) - 1 * d_oo(3,1) @ ul2(i,a,b) @ t2(i,0,a,2) @ ur1(b) - 1 * d_oo(3,0) @ ul2(i,a,b) @ t2(i,1,a,b) @ ur1(2)))  # generated
    ret["ooov"] = (i5_ov(0,3) @ d_oo(1,2) - i5_ov(1,3) @ d_oo(0,2)
                   + 1 * d_oo(2,0) @ ul1(a) @ ur1(3) @ t2s(1,a) - 1 * d_oo(2,1) @ ul1(a) @ ur1(3) @ t2s(0,a)  # generated
                   + (1/sqrt(2)) * (ul2(2,a,b) @ t2(0,1,a,3) @ ur1(b) - 1 * ul2(2,a,b) @ t2(0,1,a,b) @ ur1(3) - 1 * ul2(2,a,b) @ t2(0,1,b,3) @ ur1(a) + 1 * d_oo(2,0) @ ul2(i,a,b) @ t2(i,1,b,3) @ ur1(a) + 1 * d_oo(2,1) @ ul2(i,a,b) @ t2(i,0,a,3) @ ur1(b) + 1 * d_oo(2,0) @ ul2(i,a,b) @ t2(i,1,a,b) @ ur1(3) - 1 * d_oo(2,0) @ ul2(i,a,b) @ t2(i,1,a,3) @ ur1(b) - 1 * d_oo(2,1) @ ul2(i,a,b) @ t2(i,0,b,3) @ ur1(a) - 1 * d_oo(2,1) @ ul2(i,a,b) @ t2(i,0,a,b) @ ur1(3)))  # generated
    ret["vvvv"] =  (+ 0.5 * t2(l,m,0,d) @ t2(l,m,3,d) @ ul1(1) @ ur1(2) + 0.5 * t2(l,m,1,d) @ t2(l,m,2,d) @ ul1(0) @ ur1(3) + 0.5 * t2(l,m,a,3) @ t2(l,m,0,1) @ ul1(a) @ ur1(2) + 0.5 * t2(l,m,b,1) @ t2(l,m,2,3) @ ul1(0) @ ur1(b) - 0.5 * t2(l,m,0,d) @ t2(l,m,2,d) @ ul1(1) @ ur1(3) - 0.5 * t2(l,m,1,d) @ t2(l,m,3,d) @ ul1(0) @ ur1(2) - 0.5 * t2(l,m,a,2) @ t2(l,m,0,1) @ ul1(a) @ ur1(3) - 0.5 * t2(l,m,b,0) @ t2(l,m,2,3) @ ul1(1) @ ur1(b)  # generated
                    - 2 * ul2(i,0,1) @ ur2(i,2,3))  # generated
    
    return ret




