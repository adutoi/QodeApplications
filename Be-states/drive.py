import numpy
from qode.util.PyC import import_C, Double, BigInt

antisymm = import_C("antisymm", flags="-O3 -fopenmp")

Ddummy = numpy.zeros(1, dtype=Double.numpy)
Idummy = numpy.zeros(1, dtype=BigInt.numpy)

def test_it():
    antisymm.antisymmetrize(Ddummy,      # the input/output tensor
                            4,           # the number of tensor axes
                            5,           # the length of the axes
                            [10],        # the vector length of the "elements" of the tensor (given as a reference!)
                            0,           # for recursive use.  0 on first input
                            0,           # for recursive use.  0 on first input
                            1,           # for recursive use.  1 on first input
                            [Idummy],    # for recursive use.  NULL on first input
                            [1])         # for recursive use.  [1] on first input

test_it()









