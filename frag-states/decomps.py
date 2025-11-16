import sys
import pickle
import itertools
import numpy
import tensorly
from qode.util.PyC       import Double
from qode.math.tensornet import raw, tl_tensor

frag = sys.argv[1]

def tens_wrap(tensor):
    return tl_tensor.init(tensorly.tensor(tensor, dtype=Double.tensorly))


Be = pickle.load(open(f"rho/Be-Be_{frag}_6-31G_nth_compress.pkl", "rb"))
nstates = Be.rho["n_states"]

projC, projV = [0]*18, [0]*18
projC[0], projC[9] = 1, 1
for p in range(18):
    if projC[p]!=1:
        projV[p] = 1
projC = tens_wrap(numpy.diag(projC))
projV = tens_wrap(numpy.diag(projV))

op_loop = list(Be.rho.keys())    # because we add to the ...
for op_string in op_loop:        # ... dict while looping
    n_ops = len(op_string)
    indices = list(range(2+n_ops))
    Be.rho[op_string+"C"] = {}
    Be.rho[op_string+"V"] = {}
    if op_string in ("n_elec", "n_states"):
        pass
    else:
        for charges in Be.rho[op_string]:
            tmp = Be.rho[op_string][charges]
            for p in range(2, 2+n_ops):
                contract = list(indices)
                contract[p] = "p"
                tmp = tmp(*contract) @ projC("p", p)
            Be.rho[op_string+"C"][charges] = tmp
            tmp = Be.rho[op_string][charges]
            for p in range(2, 2+n_ops):
                contract = list(indices)
                contract[p] = "p"
                tmp = tmp(*contract) @ projV("p", p)
            Be.rho[op_string+"V"][charges] = tmp
#Be.rho["caC"]   = Be.rho["caC"  ][(0,0)][0,0,:,:]        # ok, but makes ...
#Be.rho["ccaaC"] = Be.rho["ccaaC"][(0,0)][0,0,:,:,:,:]    # ... tests slower
Be.rho["caC"]   = tens_wrap(raw(Be.rho["caC"  ][(0,0)][0,0,:,:]))
Be.rho["ccaaC"] = tens_wrap(raw(Be.rho["ccaaC"][(0,0)][0,0,:,:,:,:]))

def test_formulas(reference, formulas, *, description=None, verbose=False):
    if description:  print(description)
    print(f" i,  j: |" + "".join(f"{header:9s}|" for header,_ in formulas))
    formulas = [(header,formula()) for header,formula in formulas]
    for i in range(reference.shape[0]):
        for j in range(reference.shape[1]):
            extra = ""
            print(f"{i:2d}, {j:2d}:", end="")
            ref_val = raw(reference[i,j,...])
            ref_norm  = numpy.linalg.norm(ref_val)
            for header,formula in formulas:
                test_val  = raw(formula[i,j,...])
                test_norm = numpy.linalg.norm(test_val)
                diff_norm = numpy.linalg.norm(test_val - ref_val)
                error = 100 * diff_norm/ref_norm    
                print(f"  {error:6.2f} %", end="")
                if verbose:
                    extra += f"    {header}:  ref norm = {ref_norm}, test norm = {test_norm}, diff_norm = {diff_norm}\n"
            print()
            if verbose:  print(extra, end="")

def anti(tensor, groups):
    #
    def unique(iterable, size):
        indexable = list(iterable)
        if size==1:
            result = [[x] for x in indexable]
        else:
            result = []
            for i,x in enumerate(indexable):
                nested = unique(indexable[i+1:], size-1)
                result += [[x]+n for n in nested]
        return result
    #
    groupA, groupB = groups.pop()
    if len(groups)>0:
        tensor = anti(tensor, groups)
    if len(groupB)<len(groupA):
        groupA, groupB = groupB, groupA
    permutations = [(+1, list(range(len(tensor.shape))))]
    for perm_len in range(len(groupA)+1):
        uniqueA = unique(groupA, perm_len)
        uniqueB = unique(groupB, perm_len)
        for swapA in uniqueA:
            for swapB in uniqueB:
                permutation = list(range(len(tensor.shape)))
                for a,b in zip(swapA, swapB):
                    permutation[a] = b
                    permutation[b] = a
                permutations += [((-1)**perm_len, permutation)]
    #for sign,permutation in permutations:
    #    print(f"{sign:+d}  {permutation}")
    result = tl_tensor.zeros()    # takes its shape from summed terms
    for sign,permutation in permutations:
        result += sign * tensor(*permutation)
    return result



i, j, p, q, r, s, t, u, v, w, x, y, z = range(13)

def ccccaa():
    caC    = Be.rho["caC"]
    ccaaC  = Be.rho["ccaaC"]
    ccV    = Be.rho["ccV"  ][(-1,+1)]
    cccaV  = Be.rho["cccaV"][(-1,+1)]
    return anti(caC(p,u) @ cccaV(i,j,q,r,s,t), [((p,),(q,r,s)), ((t,),(u,))]) + anti(ccaaC(p,q,t,u) @ ccV(i,j,r,s), [((p,q),(r,s))])

def ccaaaa():
    caC    = Be.rho["caC"]
    ccaaC  = Be.rho["ccaaC"]
    aaV    = Be.rho["aaV"  ][(+1,-1)]
    caaaV  = Be.rho["caaaV"][(+1,-1)]
    return anti(caC(p,u) @ caaaV(i,j,q,r,s,t), [((p,),(q,)), ((r,s,t),(u,))]) + anti(ccaaC(p,q,t,u) @ aaV(i,j,r,s), [((r,s),(t,u))])

def ccccaaaA():
    caC    = Be.rho["caC"]
    ccaaC  = Be.rho["ccaaC"]
    ccaV   = Be.rho["ccaV"  ][(-1,0)]
    cccaaV = Be.rho["cccaaV"][(-1,0)]
    return -anti(caC(p,v) @ cccaaV(i,j,q,r,s,t,u), [((p,),(q,r,s)), ((t,u),(v,))]) + anti(ccaaC(p,q,u,v) @ ccaV(i,j,r,s,t), [((p,q),(r,s)), ((t,),(u,v))])

def ccccaaaB():
    ccaaC  = Be.rho["ccaaC"]
    ccaV   = Be.rho["ccaV"  ][(0,+1)]
    return anti(ccaaC(p,q,u,v) @ ccaV(i,j,r,s,t), [((p,q),(r,s)), ((t,),(u,v))])

def cccaaaaA():
    caC    = Be.rho["caC"]
    ccaaC  = Be.rho["ccaaC"]
    caaV   = Be.rho["caaV"  ][(0,-1)]
    ccaaaV = Be.rho["ccaaaV"][(0,-1)]
    return -anti(caC(p,v) @ ccaaaV(i,j,q,r,s,t,u), [((p,),(q,r)), ((s,t,u),(v,))]) + anti(ccaaC(p,q,u,v) @ caaV(i,j,r,s,t), [((p,q),(r,)), ((s,t),(u,v))])

def cccaaaaB():
    ccaaC  = Be.rho["ccaaC"]
    caaV   = Be.rho["caaV"  ][(+1,0)]
    return anti(ccaaC(p,q,u,v) @ caaV(i,j,r,s,t), [((p,q),(r,)), ((s,t),(u,v))])

def cccccaaa():
    ccaaC  = Be.rho["ccaaC"]
    cccaV  = Be.rho["cccaV"][(-1,+1)]
    return anti(ccaaC(p,q,v,w) @ cccaV(i,j,r,s,t,u), [((p,q),(r,s,t)), ((u,),(v,w))])

def ccccaaaaA():
    caC     = Be.rho["caC"]
    ccaaC   = Be.rho["ccaaC"]
    ccaaV   = Be.rho["ccaaV"  ][(-1,-1)]
    cccaaaV = Be.rho["cccaaaV"][(-1,-1)]
    return anti(caC(p,w) @ cccaaaV(i,j,q,r,s,t,u,v), [((p,),(q,r,s)), ((t,u,v),(w,))]) + anti(ccaaC(p,q,v,w) @ ccaaV(i,j,r,s,t,u), [((p,q),(r,s)), ((t,u),(v,w))])

def ccccaaaaB():
    ccaaC   = Be.rho["ccaaC"]
    ccaaV   = Be.rho["ccaaV"  ][(0,0)]
    return anti(ccaaC(p,q,v,w) @ ccaaV(i,j,r,s,t,u), [((p,q),(r,s)), ((t,u),(v,w))])

def cccaaaaa():
    ccaaC  = Be.rho["ccaaC"]
    caaaV  = Be.rho["caaaV"][(+1,-1)]
    return anti(ccaaC(p,q,v,w) @ caaaV(i,j,r,s,t,u), [((p,q),(r,)), ((s,t,u,),(v,w))])



if True:
    Be.rho["ccccaa"] = {}
    Be.rho["ccccaa"][(-1,+1)] = ccccaa()
    Be.rho["ccaaaa"] = {}
    Be.rho["ccaaaa"][(+1,-1)] = ccaaaa()
    Be.rho["ccccaaa"] = {}
    Be.rho["ccccaaa"][(-1,0)] = ccccaaaA()
    Be.rho["ccccaaa"][(0,+1)] = ccccaaaB()
    Be.rho["cccaaaa"] = {}
    Be.rho["cccaaaa"][(0,-1)] = cccaaaaA()
    Be.rho["cccaaaa"][(+1,0)] = cccaaaaB()
    Be.rho["cccccaaa"] = {}
    Be.rho["cccccaaa"][(-1,+1)] = cccccaaa()
    Be.rho["ccccaaaa"] = {}
    Be.rho["ccccaaaa"][(-1,-1)] = ccccaaaaA()
    Be.rho["ccccaaaa"][( 0, 0)] = ccccaaaaB()
    Be.rho["cccaaaaa"] = {}
    Be.rho["cccaaaaa"][(+1,-1)] = cccaaaaa()
    pickle.dump(Be, open(f"rho/Be-Be_{frag}_6-31G_nth_compress-factored.pkl", "wb"))


if False:
    test_formulas(
        Be.rho["ccaa"][(+1,+1)],
        [
            ("ccaa", ccaa),
        ],
        description="ccaa",
        #verbose=True
    )

if False:
    test_formulas(
        Be.rho["cccaa"][(0,+1)],
        [
            ("cccaa", cccaa),
        ],
        description="cccaa",
        #verbose=True
    )

if False:
    test_formulas(
        Be.rho["ccaaa"][(+1,0)],
        [
            ("ccaaa", ccaaa),
        ],
        description="ccaaa",
        #verbose=True
    )

if False:
    test_formulas(
        Be.rho["cccaaa"][(0,0)],
        [
            ("cccaaa", cccaaa),
        ],
        description="cccaaa",
        #verbose=True
    )
