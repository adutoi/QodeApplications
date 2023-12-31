import pickle
import numpy
#class empty(object):  pass

nstates = {-1:8, 0:11, +1:4}
A = pickle.load(open("old-data/Be631g-thresh=1e-6:4.5.pkl","rb"))
B = pickle.load(open("/scratch/adutoi/Be631g.pkl","rb"))

def AA(op, bc, kc, b, k):
    return A.rho[op][bc,kc][b][k]._raw_tensor

def BB(op, bc, kc, b, k):
    return B.rho[op][bc,kc][b][k]._raw_tensor

def diff(op, bc, kc, b, k):
    return numpy.linalg.norm(AA(op, bc, kc, b, k) - BB(op, bc, kc, b, k))

def diffs(op, bc, kc):
    for b in range(nstates[bc]):
        for k in range(nstates[kc]):
            print("{: .1e}".format(diff(op,bc,kc,b,k)), end="")
        print()
    return

def maxdiffs(op, bc, kc):
    value = 0
    for b in range(nstates[bc]):
        for k in range(nstates[kc]):
            test = diff(op,bc,kc,b,k)
            if test>value:  value = test
    return value

print("c", 0, 1)
print(maxdiffs("c", 0, 1))

print("c", -1, 0)
print(maxdiffs("c", -1, 0))

print("cc", -1, 1)
print(maxdiffs("cc", -1, 1))

print("ccca", -1, 1)
print(maxdiffs("ccca", -1, 1))

print("aa", 1, -1)
print(maxdiffs("aa", 1, -1))

print("caaa", 1, -1)
print(maxdiffs("caaa", 1, -1))

print("c", 0, 1)
print(maxdiffs("c", 0, 1))

print("c", -1, 0)
print(maxdiffs("c", -1, 0))

print("cca", 0, 1)
print(maxdiffs("cca", 0, 1))

print("cca", -1, 0)
print(maxdiffs("cca", -1, 0))

print("cccaa", 0, 1)
print(maxdiffs("cccaa", 0, 1))

print("cccaa", -1, 0)
print(maxdiffs("cccaa", -1, 0))

print("a", 1, 0)
print(maxdiffs("a", 1, 0))

print("a", 0, -1)
print(maxdiffs("a", 0, -1))

print("caa", 1, 0)
print(maxdiffs("caa", 1, 0))

print("caa", 0, -1)
print(maxdiffs("caa", 0, -1))

print("ccaaa", 1, 0)
print(maxdiffs("ccaaa", 1, 0))

print("ccaaa", 0, -1)
print(maxdiffs("ccaaa", 0, -1))

print("ca", 1, 1)
print(maxdiffs("ca", 1, 1))

print("ca", 0, 0)
print(maxdiffs("ca", 0, 0))

print("ca", -1, -1)
print(maxdiffs("ca", -1, -1))

print("ccaa", 1, 1)
print(maxdiffs("ccaa", 1, 1))

print("ccaa", 0, 0)
print(maxdiffs("ccaa", 0, 0))

print("ccaa", -1, -1)
print(maxdiffs("ccaa", -1, -1))
