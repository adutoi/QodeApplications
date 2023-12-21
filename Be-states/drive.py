import numpy
from qode.util.PyC import import_C, Double, BigInt

antisymm = import_C("antisymm", flags="-O3 -fopenmp")

Ddummy = numpy.zeros(1, dtype=Double.numpy)
Idummy = numpy.zeros(1, dtype=BigInt.numpy)

def test_it():
    antisymm.antisymmetrize(Ddummy, 5, 4, [Idummy], 1, 0, [10], 0, [1], 0)

test_it()









