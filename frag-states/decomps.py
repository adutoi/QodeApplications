import pickle
import numpy
from qode.math.tensornet import raw, np_tensor, tensor_sum
from qode.util.dynamic_array import dynamic_array

core = [slice(0,1), slice(9,10)]
valence = [slice(1,9), slice(10,18)]

Be = pickle.load(open("rho/Be-Be_0_6-31G_nth_compress.pkl","rb"))
nstates = Be.rho["n_states"]

ca_   = {}
caC   = {}
caV   = {}
ccaa_ = {}
ccaaC = {}
ccaaV = {}
cccaaa_ = {}
for i in range(nstates[0]):
    for j in range(nstates[0]):
        # np_tensor(raw()) should be unnecessary. I guess there is a bug in handling slices.
        ca_[i,j]     = raw(Be.rho["ca"    ][0,0][i,j,:,:])
        ccaa_[i,j]   = raw(Be.rho["ccaa"  ][0,0][i,j,:,:,:,:])
        cccaaa_[i,j] = np_tensor(raw(Be.rho["cccaaa"][0,0][i,j,:,:,:,:,:,:]))
        caC[i,j] = numpy.zeros_like(ca_[i,j])
        caV[i,j] = numpy.zeros_like(ca_[i,j])
        ccaaC[i,j] = numpy.zeros_like(ccaa_[i,j])
        ccaaV[i,j] = numpy.zeros_like(ccaa_[i,j])
        for p in core:
            for q in core:
                caC[i,j][p,q] = ca_[i,j][p,q]
                for r in core:
                    for s in core:
                        ccaaC[i,j][p,q,r,s] = ccaa_[i,j][p,q,r,s]
        for p in valence:
            for q in valence:
                caV[i,j][p,q] = ca_[i,j][p,q]
                for r in valence:
                    for s in valence:
                        ccaaV[i,j][p,q,r,s] = ccaa_[i,j][p,q,r,s]
        ca_[i,j] = np_tensor(ca_[i,j])
        ccaa_[i,j] = np_tensor(ccaa_[i,j])
        caC[i,j] = np_tensor(caC[i,j])
        ccaaC[i,j] = np_tensor(ccaaC[i,j])
        caV[i,j] = np_tensor(caV[i,j])
        ccaaV[i,j] = np_tensor(ccaaV[i,j])

def test_formulas(reference, formulas, nbra, nket, *, description=None, verbose=False):
    headers, formulas = zip(*formulas)
    tests = [dynamic_array(formula, (range(nbra),range(nket))) for formula in formulas]
    tests = list(zip(headers, tests))
    if description:  print(description)
    print(f" i,  j: |" + "".join(f"{header:9s}|" for header in headers))
    for i in range(nbra):
        for j in range(nket):
            extra = ""
            print(f"{i:2d}, {j:2d}:", end="")
            ref_val = raw(reference[i,j])
            ref_norm  = numpy.linalg.norm(ref_val)
            for header,test in tests:
                test_val = raw(test[i,j])
                test_norm = numpy.linalg.norm(test_val)
                diff_norm = numpy.linalg.norm(test_val - ref_val)
                error = 100 * diff_norm/ref_norm    
                print(f"  {error:6.2f} %", end="")
                if verbose:
                    extra += f"    {header}:  ref norm = {ref_norm}, test norm = {test_norm}, diff_norm = {diff_norm}\n"
            print()
            if verbose:  print(extra, end="")



def anti2(formula):
    def tensor(i, j):
        result = formula(i, j)
        return ( result(0,1,2,3) - result(1,0,2,3) - result(0,1,3,2) + result(1,0,3,2) ) / 2
    return tensor

def kernel2(ca1, ca2):
    return ca1(0,3) @ ca2(1,2)

@anti2
def basic2(i,j):
    return kernel2(ca_[i,j], ca_[i,j])

@anti2
def RI2(i,j):
    delta = np_tensor(numpy.identity(ca_[0,0].shape[0]))
    result = tensor_sum()
    for k in range(nstates[0]):
        result += kernel2(ca_[i,k], ca_[k,j])
    result -= delta(1,3) @ ca_[i,j](0,2)
    return result

@anti2
def RI2_short(i,j):
    delta = np_tensor(numpy.identity(ca_[0,0].shape[0]))
    result = tensor_sum()
    for k in range(nstates[0]):
        result += kernel2(ca_[i,k], ca_[k,j])
    return result

"""
test_formulas(
    ccaa_,
    [
        ("basic2", basic2),
        ("RI2", RI2),
        ("RI2 short", RI2_short)
    ],
    nstates[0], nstates[0],
    description="ccaa",
)
"""


def kernel3(ca, ccaa):
    return (
          ca(0,5) @ ccaa(1,2,3,4)
        - ca(0,4) @ ccaa(1,2,3,5)
        - ca(0,3) @ ccaa(1,2,5,4)
        - ca(1,5) @ ccaa(0,2,3,4)
        + ca(1,4) @ ccaa(0,2,3,5)
        + ca(1,3) @ ccaa(0,2,5,4)
        - ca(2,5) @ ccaa(1,0,3,4)
        + ca(2,4) @ ccaa(1,0,3,5)
        + ca(2,3) @ ccaa(1,0,5,4)
    ) 

def basic3(i,j):
    return kernel3(ca_[i,j], ccaa_[i,j]) / 3

def CVsep(i,j):
    return kernel3(caC[i,j], ccaaV[i,j]) + kernel3(caV[i,j], ccaaC[i,j])

def CVsep_hack(i,j):
    return kernel3(caC[i,i], ccaaV[i,j]) + kernel3(caV[i,j], ccaaC[i,i])

test_formulas(
    cccaaa_,
    [
        ("basic3", basic3),
        ("CVsep", CVsep),
        ("CVsep_hack", CVsep_hack),
    ],
    nstates[0], nstates[0],
    description="cccaaa",
    #verbose = True
)

#numpy.set_printoptions(precision=1, linewidth=500)
#print(raw(ca_[0,0]))
