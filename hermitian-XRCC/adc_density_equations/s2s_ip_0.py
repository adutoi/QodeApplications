#import numpy as np
from qode.math.tensornet import evaluate, scalar_value, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, m, a, b, c, d, e = 'ijklmabcde'        # lower the number of quotes we need to type

def s2s_2p_ip_0(d_oo, t2, t2s, t2d, vec_left, vec_right):
    ret = {}
    ur1 = tl_tensor(tl.tensor(vec_right.h.to_ndarray(), dtype=tl.float64))
    ur2 = tl_tensor(tl.tensor(vec_right.phh.to_ndarray(), dtype=tl.float64))
    ul1 = tl_tensor(tl.tensor(vec_left.h.to_ndarray(), dtype=tl.float64))
    ul2 = tl_tensor(tl.tensor(vec_left.phh.to_ndarray(), dtype=tl.float64))
    i2_ovv_r = evaluate(ur1(j) @ t2(j,0,1,2))
    i3_ovv_l = evaluate(ul1(i) @ t2(i,0,1,2))
    i2_ovv_rd = evaluate(ur1(j) @ t2d(j,0,1,2))
    i3_ovv_ld = evaluate(ul1(i) @ t2d(i,0,1,2))
    ur1_sqrt2 = evaluate(sqrt(2) * ur1)
    ul1_sqrt2 = evaluate(sqrt(2) * ul1)
    i4_ov = evaluate(ul2(0,j,1) @ ur1_sqrt2(j))
    i5_ov = evaluate(ul1_sqrt2(i) @ ur2(i,0,1))
    t2_squared = scalar_value(t2(l,m,d,e) @ t2(l,m,d,e))


    ret["oooo"] = (
        ur1(0) @ ul1(2) @ d_oo(1,3)  # zeroth order
        - ur1(1) @ ul1(2) @ d_oo(0,3)  # zeroth order
        - ur1(0) @ ul1(3) @ d_oo(1,2)  # zeroth order
        + ur1(1) @ ul1(3) @ d_oo(0,2)  # zeroth order
        + 0.5 * t2(0,l,d,e) @ t2(3,l,d,e) @ ul1(2) @ ur1(1) + 0.5 * t2(1,l,d,e) @ t2(2,l,d,e) @ ul1(3) @ ur1(0) + 0.5 * t2(j,3,d,e) @ t2(0,1,d,e) @ ul1(2) @ ur1(j) + 0.5 * t2(i,1,d,e) @ t2(2,3,d,e) @ ul1(i) @ ur1(0) - 0.5 * t2(0,l,d,e) @ t2(2,l,d,e) @ ul1(3) @ ur1(1) - 0.5 * t2(1,l,d,e) @ t2(3,l,d,e) @ ul1(2) @ ur1(0) - 0.5 * t2(j,2,d,e) @ t2(0,1,d,e) @ ul1(3) @ ur1(j) - 0.5 * t2(i,0,d,e) @ t2(2,3,d,e) @ ul1(i) @ ur1(1) + 0.5 * d_oo(3,0) @ t2(j,l,d,e) @ t2(1,l,d,e) @ ul1(2) @ ur1(j) + 0.5 * d_oo(2,1) @ t2(j,l,d,e) @ t2(0,l,d,e) @ ul1(3) @ ur1(j) + 0.5 * d_oo(3,0) @ t2(i,l,d,e) @ t2(2,l,d,e) @ ul1(i) @ ur1(1) + 0.5 * d_oo(2,1) @ t2(i,l,d,e) @ t2(3,l,d,e) @ ul1(i) @ ur1(0) + 0.5 * d_oo(3,0) @ t2(i,1,d,e) @ t2(j,2,d,e) @ ul1(i) @ ur1(j) + 0.5 * d_oo(2,1) @ t2(i,0,d,e) @ t2(j,3,d,e) @ ul1(i) @ ur1(j) - 0.5 * d_oo(2,0) @ t2(j,l,d,e) @ t2(1,l,d,e) @ ul1(3) @ ur1(j) - 0.5 * d_oo(3,1) @ t2(j,l,d,e) @ t2(0,l,d,e) @ ul1(2) @ ur1(j) - 0.5 * d_oo(3,1) @ t2(i,l,d,e) @ t2(2,l,d,e) @ ul1(i) @ ur1(0) - 0.5 * d_oo(2,0) @ t2(i,l,d,e) @ t2(3,l,d,e) @ ul1(i) @ ur1(1) - 0.5 * d_oo(2,0) @ t2(i,1,d,e) @ t2(j,3,d,e) @ ul1(i) @ ur1(j) - 0.5 * d_oo(3,1) @ t2(i,0,d,e) @ t2(j,2,d,e) @ ul1(i) @ ur1(j)  # generated
        + 0.25 * t2_squared * ul1(2) @ ur1(0) @ d_oo(1,3) - 0.25 * t2_squared * ul1(3) @ ur1(0) @ d_oo(1,2) - 0.25 * t2_squared * ul1(2) @ ur1(1) @ d_oo(0,3) + 0.25 * t2_squared * ul1(3) @ ur1(1) @ d_oo(0,2)
        - 2 * ul2(2,3,a) @ ur2(0,1,a) + 0.5 * d_oo(3,1) @ ul2(i,2,a) @ ur2(i,0,a) + 0.5 * d_oo(2,0) @ ul2(i,3,a) @ ur2(i,1,a) + 0.5 * d_oo(3,1) @ ul2(j,2,a) @ ur2(j,0,a) + 0.5 * d_oo(2,0) @ ul2(j,3,a) @ ur2(j,1,a) + 1 * d_oo(3,1) @ ul2(2,l,a) @ ur2(0,l,a) + 1 * d_oo(2,0) @ ul2(3,l,a) @ ur2(1,l,a) - 0.5 * d_oo(3,0) @ ul2(i,2,a) @ ur2(i,1,a) - 0.5 * d_oo(2,1) @ ul2(i,3,a) @ ur2(i,0,a) - 0.5 * d_oo(3,0) @ ul2(j,2,a) @ ur2(j,1,a) - 0.5 * d_oo(2,1) @ ul2(j,3,a) @ ur2(j,0,a) - 1 * d_oo(3,0) @ ul2(2,l,a) @ ur2(1,l,a) - 1 * d_oo(2,1) @ ul2(3,l,a) @ ur2(0,l,a)  # generated
    )

    ret["vvoo"] = (#i1 * t2(2,3,0,1) 
                   ul1(3) @ i2_ovv_r(2,0,1) - ul1(2) @ i2_ovv_r(3,0,1)  # first order
                   #+ i1 * t2d(2,3,0,1) 
                   + ul1(3) @ i2_ovv_rd(2,0,1) - ul1(2) @ i2_ovv_rd(3,0,1))
    ret["oovv"] = (#i1 * t2(0,1,2,3) 
                   ur1(1) @ i3_ovv_l(0,2,3) - ur1(0) @ i3_ovv_l(1,2,3)  # first order
                   #+ i1 * t2d(0,1,2,3) 
                   + ur1(1) @ i3_ovv_ld(0,2,3) - ur1(0) @ i3_ovv_ld(1,2,3))
    
    ret["vooo"] = (ul2(2,3,0) @ ur1_sqrt2(1) + i4_ov(3,0) @ d_oo(1,2) - i4_ov(2,0) @ d_oo(1,3)
                   + 1 * ul1(3) @ ur1(1) @ t2s(2,0) - 1 * ul1(2) @ ur1(1) @ t2s(3,0) + 1 * d_oo(3,1) @ ul1(2) @ ur1(j) @ t2s(j,0) - 1 * d_oo(2,1) @ ul1(3) @ ur1(j) @ t2s(j,0)  # generated
                   + (1/sqrt(2)) * (ur2(j,1,a) @ t2(j,3,a,0) @ ul1(2) + 1 * ur2(k,1,a) @ t2(k,3,a,0) @ ul1(2) - 1 * ur2(j,1,a) @ t2(j,2,a,0) @ ul1(3) - 1 * ur2(k,1,a) @ t2(k,2,a,0) @ ul1(3) - 2 * ur2(i,1,a) @ t2(2,3,a,0) @ ul1(i) + 1 * d_oo(2,1) @ ur2(i,j,a) @ t2(j,3,a,0) @ ul1(i) + 1 * d_oo(2,1) @ ur2(i,k,a) @ t2(k,3,a,0) @ ul1(i) + 1 * d_oo(2,1) @ ur2(j,k,a) @ t2(j,k,a,0) @ ul1(3) - 1 * d_oo(3,1) @ ur2(i,j,a) @ t2(j,2,a,0) @ ul1(i) - 1 * d_oo(3,1) @ ur2(i,k,a) @ t2(k,2,a,0) @ ul1(i) - 1 * d_oo(3,1) @ ur2(j,k,a) @ t2(j,k,a,0) @ ul1(2)))  # generated
    ret["ovoo"] = (- ul2(2,3,1) @ ur1_sqrt2(0) - i4_ov(3,1) @ d_oo(0,2) + i4_ov(2,1) @ d_oo(0,3)
                   + 1 * ul1(2) @ ur1(0) @ t2s(3,1) - 1 * ul1(3) @ ur1(0) @ t2s(2,1) + 1 * d_oo(2,0) @ ul1(3) @ ur1(j) @ t2s(j,1) - 1 * d_oo(3,0) @ ul1(2) @ ur1(j) @ t2s(j,1)  # generated
                   + (1/sqrt(2)) * (ur2(j,0,a) @ t2(j,2,a,1) @ ul1(3) + 1 * ur2(k,0,a) @ t2(k,2,a,1) @ ul1(3) + 2 * ur2(i,0,a) @ t2(2,3,a,1) @ ul1(i) - 1 * ur2(j,0,a) @ t2(j,3,a,1) @ ul1(2) - 1 * ur2(k,0,a) @ t2(k,3,a,1) @ ul1(2) + 1 * d_oo(3,0) @ ur2(i,j,a) @ t2(j,2,a,1) @ ul1(i) + 1 * d_oo(3,0) @ ur2(i,k,a) @ t2(k,2,a,1) @ ul1(i) + 1 * d_oo(3,0) @ ur2(j,k,a) @ t2(j,k,a,1) @ ul1(2) - 1 * d_oo(2,0) @ ur2(i,j,a) @ t2(j,3,a,1) @ ul1(i) - 1 * d_oo(2,0) @ ur2(i,k,a) @ t2(k,3,a,1) @ ul1(i) - 1 * d_oo(2,0) @ ur2(j,k,a) @ t2(j,k,a,1) @ ul1(3)))  # generated
    ret["oovo"] = (ul1_sqrt2(3) @ ur2(0,1,2) + i5_ov(0,2) @ d_oo(1,3) - i5_ov(1,2) @ d_oo(0,3)
                   + 1 * ul1(3) @ ur1(1) @ t2s(0,2) - 1 * ul1(3) @ ur1(0) @ t2s(1,2) + 1 * d_oo(3,1) @ ul1(i) @ ur1(0) @ t2s(i,2) - 1 * d_oo(3,0) @ ul1(i) @ ur1(1) @ t2s(i,2)  # generated
                   + (1/sqrt(2)) * (ul2(i,3,a) @ t2(i,1,a,2) @ ur1(0) + 1 * ul2(j,3,a) @ t2(j,1,a,2) @ ur1(0) - 1 * ul2(i,3,a) @ t2(0,1,a,2) @ ur1(i) - 1 * ul2(i,3,a) @ t2(i,0,a,2) @ ur1(1) - 1 * ul2(j,3,a) @ t2(0,1,a,2) @ ur1(j) - 1 * ul2(j,3,a) @ t2(j,0,a,2) @ ur1(1) + 1 * d_oo(3,0) @ ul2(i,j,a) @ t2(j,1,a,2) @ ur1(i) + 1 * d_oo(3,1) @ ul2(i,j,a) @ t2(i,0,a,2) @ ur1(j) + 1 * d_oo(3,0) @ ul2(i,j,a) @ t2(i,j,a,2) @ ur1(1) - 1 * d_oo(3,0) @ ul2(i,j,a) @ t2(i,1,a,2) @ ur1(j) - 1 * d_oo(3,1) @ ul2(i,j,a) @ t2(i,j,a,2) @ ur1(0) - 1 * d_oo(3,1) @ ul2(i,j,a) @ t2(j,0,a,2) @ ur1(i)))  # generated
    ret["ooov"] = (- ul1_sqrt2(2) @ ur2(0,1,3) - i5_ov(0,3) @ d_oo(1,2) + i5_ov(1,3) @ d_oo(0,2)
                   + 1 * ul1(2) @ ur1(0) @ t2s(1,3) - 1 * ul1(2) @ ur1(1) @ t2s(0,3) + 1 * d_oo(2,0) @ ul1(i) @ ur1(1) @ t2s(i,3) - 1 * d_oo(2,1) @ ul1(i) @ ur1(0) @ t2s(i,3)  # generated
                   + (1/sqrt(2)) * (ul2(i,2,a) @ t2(0,1,a,3) @ ur1(i) + 1 * ul2(i,2,a) @ t2(i,0,a,3) @ ur1(1) + 1 * ul2(j,2,a) @ t2(0,1,a,3) @ ur1(j) + 1 * ul2(j,2,a) @ t2(j,0,a,3) @ ur1(1) - 1 * ul2(i,2,a) @ t2(i,1,a,3) @ ur1(0) - 1 * ul2(j,2,a) @ t2(j,1,a,3) @ ur1(0) + 1 * d_oo(2,0) @ ul2(i,j,a) @ t2(i,1,a,3) @ ur1(j) + 1 * d_oo(2,1) @ ul2(i,j,a) @ t2(i,j,a,3) @ ur1(0) + 1 * d_oo(2,1) @ ul2(i,j,a) @ t2(j,0,a,3) @ ur1(i) - 1 * d_oo(2,0) @ ul2(i,j,a) @ t2(j,1,a,3) @ ur1(i) - 1 * d_oo(2,1) @ ul2(i,j,a) @ t2(i,0,a,3) @ ur1(j) - 1 * d_oo(2,0) @ ul2(i,j,a) @ t2(i,j,a,3) @ ur1(1)))  # generated
    ret["ovov"] =  (0.5 * t2(l,m,1,d) @ t2(l,m,3,d) @ ul1(2) @ ur1(0) - 1 * t2(j,l,1,d) @ t2(0,l,3,d) @ ul1(2) @ ur1(j) - 1 * t2(i,l,3,d) @ t2(2,l,1,d) @ ul1(i) @ ur1(0) - 1 * t2(i,0,3,d) @ t2(j,2,1,d) @ ul1(i) @ ur1(j) + 1 * d_oo(2,0) @ t2(i,l,3,d) @ t2(j,l,1,d) @ ul1(i) @ ur1(j)  # generated
                    + 0.5 * ul2(i,2,1) @ ur2(i,0,3) + 0.5 * ul2(j,2,1) @ ur2(j,0,3) + 1 * ul2(2,l,1) @ ur2(0,l,3) - 0.5 * d_oo(2,0) @ ul2(i,l,1) @ ur2(i,l,3) - 0.5 * d_oo(2,0) @ ul2(j,l,1) @ ur2(j,l,3))  # generated
    ret["ovvo"] =  (t2(j,l,1,d) @ t2(0,l,2,d) @ ul1(3) @ ur1(j) + 1 * t2(i,l,2,d) @ t2(3,l,1,d) @ ul1(i) @ ur1(0) + 1 * t2(i,0,2,d) @ t2(j,3,1,d) @ ul1(i) @ ur1(j) - 0.5 * t2(l,m,1,d) @ t2(l,m,2,d) @ ul1(3) @ ur1(0) - 1 * d_oo(3,0) @ t2(i,l,2,d) @ t2(j,l,1,d) @ ul1(i) @ ur1(j)  # generated
                    - 0.5 * ul2(i,3,1) @ ur2(i,0,2) - 0.5 * ul2(j,3,1) @ ur2(j,0,2) - 1 * ul2(3,l,1) @ ur2(0,l,2) + 0.5 * d_oo(3,0) @ ul2(i,l,1) @ ur2(i,l,2) + 0.5 * d_oo(3,0) @ ul2(j,l,1) @ ur2(j,l,2))  # generated
    ret["voov"] =  (t2(j,l,0,d) @ t2(1,l,3,d) @ ul1(2) @ ur1(j) + 1 * t2(i,l,3,d) @ t2(2,l,0,d) @ ul1(i) @ ur1(1) + 1 * t2(i,1,3,d) @ t2(j,2,0,d) @ ul1(i) @ ur1(j)- 0.5 * t2(l,m,0,d) @ t2(l,m,3,d) @ ul1(2) @ ur1(1) - 1 * d_oo(2,1) @ t2(i,l,3,d) @ t2(j,l,0,d) @ ul1(i) @ ur1(j)  # generated
                    - 0.5 * ul2(i,2,0) @ ur2(i,1,3) - 0.5 * ul2(j,2,0) @ ur2(j,1,3) - 1 * ul2(2,l,0) @ ur2(1,l,3) + 0.5 * d_oo(2,1) @ ul2(i,l,0) @ ur2(i,l,3) + 0.5 * d_oo(2,1) @ ul2(j,l,0) @ ur2(j,l,3))  # generated
    ret["vovo"] =  (0.5 * t2(l,m,0,d) @ t2(l,m,2,d) @ ul1(3) @ ur1(1) - 1 * t2(j,l,0,d) @ t2(1,l,2,d) @ ul1(3) @ ur1(j) - 1 * t2(i,l,2,d) @ t2(3,l,0,d) @ ul1(i) @ ur1(1) - 1 * t2(i,1,2,d) @ t2(j,3,0,d) @ ul1(i) @ ur1(j) + 1 * d_oo(3,1) @ t2(i,l,2,d) @ t2(j,l,0,d) @ ul1(i) @ ur1(j)  # generated
                    + 0.5 * ul2(i,3,0) @ ur2(i,1,2) + 0.5 * ul2(j,3,0) @ ur2(j,1,2) + 1 * ul2(3,l,0) @ ur2(1,l,2) - 0.5 * d_oo(3,1) @ ul2(i,l,0) @ ur2(i,l,2) - 0.5 * d_oo(3,1) @ ul2(j,l,0) @ ur2(j,l,2))  # generated
    ret["vvvv"] =  t2(i,l,2,3) @ t2(j,l,0,1) @ ul1(i) @ ur1(j)  # generated
    ret["ovvv"] =  ul2(i,j,1) @ t2(i,j,2,3) @ ur1(0) + 1 * ul2(i,j,1) @ t2(j,0,2,3) @ ur1(i) - 1 * ul2(i,j,1) @ t2(i,0,2,3) @ ur1(j)  # generated
    ret["vovv"] =  ul2(i,j,0) @ t2(i,1,2,3) @ ur1(j) - 1 * ul2(i,j,0) @ t2(i,j,2,3) @ ur1(1) - 1 * ul2(i,j,0) @ t2(j,1,2,3) @ ur1(i)  # generated
    ret["vvov"] =  ur2(i,j,3) @ t2(j,2,0,1) @ ul1(i) + 1 * ur2(i,k,3) @ t2(k,2,0,1) @ ul1(i) + 1 * ur2(j,k,3) @ t2(j,k,0,1) @ ul1(2)  # generated
    ret["vvvo"] =  - 1 * ur2(i,j,2) @ t2(j,3,0,1) @ ul1(i) - 1 * ur2(i,k,2) @ t2(k,3,0,1) @ ul1(i) - 1 * ur2(j,k,2) @ t2(j,k,0,1) @ ul1(3)  # generated
    
    return ret




