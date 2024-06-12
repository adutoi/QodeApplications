from util import *

Be = pickle.load(open("atomic_states/rho/Be631g.pkl","rb"))

for string in ["aa", "caaa", "a", "caa", "ccaaa", "ca", "ccaa", "c", "cca", "cccaa", "cc", "ccca"]:
    for charges in Be.rho[string]:
        I, J = set(), set()
        for i,j in Be.rho[string][charges]:
            I |= {i}
            J |= {j}
        num_I, num_J = len(I), len(J)
        shape = [num_I, num_J] + list(Be.rho[string][charges][0,0].shape)
        temp = tensorly.zeros(shape, dtype=tensorly.float64)
        for i in range(num_I):
            for j in range(num_J):
                temp[i,j,...] = raw(Be.rho[string][charges][i,j])
        Be.rho[string][charges] = tl_tensor(temp)

pickle.dump(Be, open("atomic_states/rho/Be631g-tens.pkl", "wb"))
