#import numpy as np
from qode.math.tensornet import scalar_value

i, j, k, l, m, a, b, c, d, e = 'ijklmabcde'        # lower the number of quotes we need to type

def gs_2p_pp(d_oo, t2, t2s, t2d):
    ret = {}
    t2_squared = scalar_value(t2(l,m,d,e) @ t2(l,m,d,e))

    ret["oooo"] = (d_oo(0,3) @ d_oo(1,2) - d_oo(1,3) @ d_oo(0,2)  # zeroth order
                   + 0.25 * t2_squared * d_oo(0,3) @ d_oo(1,2) - 0.25 * t2_squared * d_oo(1,3) @ d_oo(0,2))

    ret["vvoo"] = t2(2,3,0,1) + t2d(2,3,0,1)
    ret["oovv"] = t2(0,1,2,3) + t2d(0,1,2,3)

    ret["ovov"] =  + 1 * t2(0,l,3,d) @ t2(2,l,1,d) - 0.5 * d_oo(2,0) @ t2(l,m,1,d) @ t2(l,m,3,d)  # generated
    ret["ovvo"] =  - 1 * t2(0,l,2,d) @ t2(3,l,1,d) + 0.5 * d_oo(3,0) @ t2(l,m,1,d) @ t2(l,m,2,d)  # generated
    ret["voov"] =  - 1 * t2(1,l,3,d) @ t2(2,l,0,d) + 0.5 * d_oo(2,1) @ t2(l,m,0,d) @ t2(l,m,3,d)  # generated
    ret["vovo"] =  + 1 * t2(1,l,2,d) @ t2(3,l,0,d) - 0.5 * d_oo(3,1) @ t2(l,m,0,d) @ t2(l,m,2,d)  # generated
    ret["vvvv"] =  - 0.5 * t2(l,m,0,1) @ t2(l,m,2,3)  # generated
    ret["vooo"] = t2s(3,0) @ d_oo(1,2) - t2s(2,0) @ d_oo(1,3)
    ret["ovoo"] = - t2s(3,1) @ d_oo(0,2) + t2s(2,1) @ d_oo(0,3)
    ret["oovo"] = - t2s(0,2) @ d_oo(1,3) + t2s(1,2) @ d_oo(0,3)
    ret["ooov"] = t2s(0,3) @ d_oo(1,2) - t2s(1,3) @ d_oo(0,2)
    return ret



