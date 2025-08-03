#import numpy as np
from qode.math.tensornet import evaluate, tl_tensor
from math import sqrt
import tensorly as tl

i, j, k, l, m, a, b, c, d, e = 'ijklmabcde'        # lower the number of quotes we need to type

def tdm_2p_pp(d_oo, t2, t2s, t2d, t2t, vec):
    ret = {}
    u1 = tl_tensor(tl.tensor(vec.ph.to_ndarray(), dtype=tl.float64))
    u2 = tl_tensor(tl.tensor(vec.pphh.to_ndarray(), dtype=tl.float64))


    ret["vooo"] = (d_oo(1,2) @ u1(3,0) - d_oo(1,3) @ u1(2,0)  # zeroth order
                   + 1 * t2(1,l,a,d) @ t2(2,l,0,d) @ u1(3,a) + 0.5 * t2(1,l,d,e) @ t2(3,l,d,e) @ u1(2,0) + 1 * t2(i,1,a,d) @ t2(2,3,0,d) @ u1(i,a) - 1 * t2(1,l,a,d) @ t2(3,l,0,d) @ u1(2,a) - 0.5 * t2(1,l,d,e) @ t2(2,l,d,e) @ u1(3,0) - 0.5 * t2(i,1,d,e) @ t2(2,3,d,e) @ u1(i,0) + 0.25 * d_oo(3,1) @ t2(i,l,d,e) @ t2(2,l,d,e) @ u1(i,0) + 0.5 * d_oo(2,1) @ t2(i,l,a,d) @ t2(3,l,0,d) @ u1(i,a) + 0.25 * d_oo(3,1) @ t2(l,m,a,d) @ t2(l,m,0,d) @ u1(2,a) - 0.5 * d_oo(3,1) @ t2(i,l,a,d) @ t2(2,l,0,d) @ u1(i,a) - 0.25 * d_oo(2,1) @ t2(i,l,d,e) @ t2(3,l,d,e) @ u1(i,0) - 0.25 * d_oo(2,1) @ t2(l,m,a,d) @ t2(l,m,0,d) @ u1(3,a)  # generated
                   )
    ret["ovoo"] = (- d_oo(0,2) @ u1(3,1) + d_oo(0,3) @ u1(2,1)  # zeroth order
                   + 1 * t2(0,l,a,d) @ t2(3,l,1,d) @ u1(2,a) + 0.5 * t2(0,l,d,e) @ t2(2,l,d,e) @ u1(3,1) + 0.5 * t2(i,0,d,e) @ t2(2,3,d,e) @ u1(i,1) - 1 * t2(0,l,a,d) @ t2(2,l,1,d) @ u1(3,a) - 0.5 * t2(0,l,d,e) @ t2(3,l,d,e) @ u1(2,1) - 1 * t2(i,0,a,d) @ t2(2,3,1,d) @ u1(i,a) + 0.5 * d_oo(3,0) @ t2(i,l,a,d) @ t2(2,l,1,d) @ u1(i,a) + 0.25 * d_oo(2,0) @ t2(i,l,d,e) @ t2(3,l,d,e) @ u1(i,1) + 0.25 * d_oo(2,0) @ t2(l,m,a,d) @ t2(l,m,1,d) @ u1(3,a) - 0.25 * d_oo(3,0) @ t2(i,l,d,e) @ t2(2,l,d,e) @ u1(i,1) - 0.5 * d_oo(2,0) @ t2(i,l,a,d) @ t2(3,l,1,d) @ u1(i,a) - 0.25 * d_oo(3,0) @ t2(l,m,a,d) @ t2(l,m,1,d) @ u1(2,a)  # generated
                   )
    
    ret["vvov"] =  t2(i,l,3,d) @ t2(2,l,1,d) @ u1(i,0) + 0.5 * t2(l,m,0,d) @ t2(l,m,3,d) @ u1(2,1) + 0.5 * t2(l,m,a,3) @ t2(l,m,0,1) @ u1(2,a) - 1 * t2(i,l,3,d) @ t2(2,l,0,d) @ u1(i,1) - 0.5 * t2(l,m,1,d) @ t2(l,m,3,d) @ u1(2,0) - 1 * t2(i,l,a,3) @ t2(2,l,0,1) @ u1(i,a)  # generated
    ret["vvvo"] =  t2(i,l,2,d) @ t2(3,l,0,d) @ u1(i,1) + 0.5 * t2(l,m,1,d) @ t2(l,m,2,d) @ u1(3,0) + 1 * t2(i,l,a,2) @ t2(3,l,0,1) @ u1(i,a) - 1 * t2(i,l,2,d) @ t2(3,l,1,d) @ u1(i,0) - 0.5 * t2(l,m,0,d) @ t2(l,m,2,d) @ u1(3,1) - 0.5 * t2(l,m,a,2) @ t2(l,m,0,1) @ u1(3,a)  # generated

    # first order
    i1_ov = evaluate(t2(i,0,a,1) @ u1(i,a))
    i1_ov_d = evaluate(t2d(i,0,a,1) @ u1(i,a))

    ret["oovo"] = (- i1_ov(1,2) @ d_oo(0,3) + i1_ov(0,2) @ d_oo(1,3) + t2(0,1,a,2) @ u1(3,a)
                   - i1_ov_d(1,2) @ d_oo(0,3) + i1_ov_d(0,2) @ d_oo(1,3) + t2d(0,1,a,2) @ u1(3,a))
    ret["ooov"] = (i1_ov(1,3) @ d_oo(0,2) - i1_ov(0,3) @ d_oo(1,2) - t2(0,1,a,3) @ u1(2,a)
                   + i1_ov_d(1,3) @ d_oo(0,2) - i1_ov_d(0,3) @ d_oo(1,2) - t2d(0,1,a,3) @ u1(2,a))
    ret["ovvv"] = (- t2(i,0,2,3) @ u1(i,1)
                   - t2d(i,0,2,3) @ u1(i,1))
    ret["vovv"] = (t2(i,1,2,3) @ u1(i,0)
                   + t2d(i,1,2,3) @ u1(i,0))
    
    ret["vvoo"] = 2 * u2(2,3,0,1)

    ret["oooo"] =  (d_oo(3,1) @ u1(2,a) @ t2s(0,a) + 1 * d_oo(2,0) @ u1(3,a) @ t2s(1,a) - 1 * d_oo(3,0) @ u1(2,a) @ t2s(1,a) - 1 * d_oo(2,1) @ u1(3,a) @ t2s(0,a)  # generated
                    - 1 * u2(2,3,a,b) @ t2(0,1,a,b) + 0.5 * d_oo(2,0) @ u2(i,3,a,b) @ t2(i,1,a,b) + 0.5 * d_oo(2,0) @ u2(j,3,a,b) @ t2(j,1,a,b) + 0.5 * d_oo(3,1) @ u2(i,2,a,b) @ t2(i,0,a,b) + 0.5 * d_oo(3,1) @ u2(j,2,a,b) @ t2(j,0,a,b) - 0.5 * d_oo(3,0) @ u2(i,2,a,b) @ t2(i,1,a,b) - 0.5 * d_oo(3,0) @ u2(j,2,a,b) @ t2(j,1,a,b) - 0.5 * d_oo(2,1) @ u2(i,3,a,b) @ t2(i,0,a,b) - 0.5 * d_oo(2,1) @ u2(j,3,a,b) @ t2(j,0,a,b)  # generated
                    )
    ret["ovov"] =  (u1(2,1) @ t2s(0,3) - 1 * d_oo(2,0) @ u1(i,1) @ t2s(i,3)  # generated
                    + 0.5 * u2(i,2,a,1) @ t2(i,0,a,3) + 0.5 * u2(i,2,b,1) @ t2(i,0,b,3) + 0.5 * u2(j,2,a,1) @ t2(j,0,a,3) + 0.5 * u2(j,2,b,1) @ t2(j,0,b,3) - 0.5 * d_oo(2,0) @ u2(i,j,a,1) @ t2(i,j,a,3) - 0.5 * d_oo(2,0) @ u2(i,j,b,1) @ t2(i,j,b,3)  # generated
                    )
    ret["ovvo"] =  (- 1 * u1(3,1) @ t2s(0,2) + 1 * d_oo(3,0) @ u1(i,1) @ t2s(i,2)  # generated
                    - 0.5 * u2(i,3,a,1) @ t2(i,0,a,2) - 0.5 * u2(i,3,b,1) @ t2(i,0,b,2) - 0.5 * u2(j,3,a,1) @ t2(j,0,a,2) - 0.5 * u2(j,3,b,1) @ t2(j,0,b,2) + 0.5 * d_oo(3,0) @ u2(i,j,a,1) @ t2(i,j,a,2) + 0.5 * d_oo(3,0) @ u2(i,j,b,1) @ t2(i,j,b,2)  # generated
                    )
    ret["voov"] =  (- 1 * u1(2,0) @ t2s(1,3) + 1 * d_oo(2,1) @ u1(i,0) @ t2s(i,3)  # generated
                    - 0.5 * u2(i,2,a,0) @ t2(i,1,a,3) - 0.5 * u2(i,2,b,0) @ t2(i,1,b,3) - 0.5 * u2(j,2,a,0) @ t2(j,1,a,3) - 0.5 * u2(j,2,b,0) @ t2(j,1,b,3) + 0.5 * d_oo(2,1) @ u2(i,j,a,0) @ t2(i,j,a,3) + 0.5 * d_oo(2,1) @ u2(i,j,b,0) @ t2(i,j,b,3)  # generated
                    )
    ret["vovo"] =  (u1(3,0) @ t2s(1,2) - 1 * d_oo(3,1) @ u1(i,0) @ t2s(i,2)  # generated
                    + 0.5 * u2(i,3,a,0) @ t2(i,1,a,2) + 0.5 * u2(i,3,b,0) @ t2(i,1,b,2) + 0.5 * u2(j,3,a,0) @ t2(j,1,a,2) + 0.5 * u2(j,3,b,0) @ t2(j,1,b,2) - 0.5 * d_oo(3,1) @ u2(i,j,a,0) @ t2(i,j,a,2) - 0.5 * d_oo(3,1) @ u2(i,j,b,0) @ t2(i,j,b,2)  # generated
                    )
    ret["vvoo"] =  u1(2,1) @ t2s(3,0) + 1 * u1(3,0) @ t2s(2,1) - 1 * u1(2,0) @ t2s(3,1) - 1 * u1(3,1) @ t2s(2,0)  # generated
    ret["oovv"] = - u1(i,a) @ t2t(i,0,1,a,2,3)
    ret["vvvv"] =  - 1 * u2(i,j,0,1) @ t2(i,j,2,3)  # generated
    
    return ret



