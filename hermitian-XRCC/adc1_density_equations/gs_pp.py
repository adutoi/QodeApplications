#import numpy as np
#from qode.math.tensornet import np_tensor, evaluate, scalar_value

i, j, k, l, a, b, c, d = 'ijklabcd'        # lower the number of quotes we need to type

def gs_2p_pp(d_oo, t2):
    ret = {}

    # zeroth order
    ret["oooo"] = d_oo(0,3) @ d_oo(1,2) - d_oo(1,3) @ d_oo(0,2)

    # first order
    ret["vvoo"] = t2(2,3,0,1)
    ret["oovv"] = t2(0,1,2,3)
    return ret



