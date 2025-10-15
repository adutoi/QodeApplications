import pickle
import itertools
import numpy
import tensorly
from qode.util.PyC       import Double
from qode.math.tensornet import raw, tl_tensor, tensor_sum

def tens_wrap(tensor):
    return tl_tensor(tensorly.tensor(tensor, dtype=Double.tensorly))


Be = pickle.load(open("rho/Be-Be_0_6-31G_nth_compress.pkl","rb"))
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

def anti(tensor, groupA, groupB):
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
    result = tensor_sum()
    for sign,permutation in permutations:
        result += sign * tensor(*permutation)
    return result



def delta(dim):
    return tens_wrap(numpy.identity(dim))

def ccaa():
    caC   = Be.rho["caC"]
    ccaaC = Be.rho["ccaaC"]
    caV   = Be.rho["caV"  ][(+1,+1)]
    dim = caV.shape[0]
    return anti(anti(caC(2,5) @ caV(0,1,3,4), (2,),(3,)), (4,),(5,))  +  delta(dim)(0,1) @ ccaaC(2,3,4,5)

def cccaa():
    caC   = Be.rho["caC"]
    ccaaC = Be.rho["ccaaC"]
    cV    = Be.rho["cV"   ][(0,+1)]
    ccaV  = Be.rho["ccaV" ][(0,+1)]
    return -anti(anti(caC(2,6) @ ccaV(0,1,3,4,5), (2,),(3,4)), (5,),(6,)) + anti(ccaaC(2,3,5,6) @ cV(0,1,4), (2,3),(4,))

def cccaaa():
    caC   = Be.rho["caC"]
    ccaaC = Be.rho["ccaaC"]
    caV   = Be.rho["caV"  ][(0,0)]
    ccaaV = Be.rho["ccaaV"][(0,0)]
    return anti(anti(caC(2,7) @ ccaaV(0,1,3,4,5,6), (2,),(3,4)), (5,6),(7,)) + anti(anti(ccaaC(2,3,6,7) @ caV(0,1,4,5), (2,3),(4,)), (5,),(6,7))



test_formulas(
    Be.rho["ccaa"][(+1,+1)],
    [
        ("ccaa", ccaa),
    ],
    description="ccaa",
    #verbose=True
)

test_formulas(
    Be.rho["cccaa"][(0,+1)],
    [
        ("cccaa", cccaa),
    ],
    description="cccaa",
    #verbose=True
)

test_formulas(
    Be.rho["cccaaa"][(0,0)],
    [
        ("cccaaa", cccaaa),
    ],
    description="cccaaa",
    #verbose=True
)
