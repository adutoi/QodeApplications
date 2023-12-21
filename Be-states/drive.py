import numpy
from qode.util.PyC import import_C, Double, BigInt

antisymm = import_C("antisymm", flags="-O3 -fopenmp")

Ddummy = numpy.zeros(1, dtype=Double.numpy)

def test_it():
    antisymm.antisymmetrize(Ddummy,      # the input/output tensor
                            5,           # the length of the axes
                            3, 2)        # the vector length of the "elements" of the tensor (given as a reference!)

test_it()









