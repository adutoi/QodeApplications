from   get_ints import get_ints
from   get_xr_result import get_xr_states, get_xr_H
from qode.math.tensornet import raw, tl_tensor

#import torch
import numpy as np
import tensorly as tl
import pickle

#torch.set_num_threads(4)
#tl.set_backend("pytorch")
#tl.set_backend("numpy")

# choose which density routine to use in the following
# old and fast
#from   build_fci_states_fast import get_fci_states
#import densities_old as densities

# newer, more flexible, but somehow slower version
from   build_fci_states import get_fci_states
import densities


def optimize_states(displacement, max_iter):
    ######################################################
    # Initialize integrals and densities
    ######################################################

    n_frag       = 2
    displacement = displacement
    project_core = True
    monomer_charges = [[0, +1, -1], [0, +1, -1]]
    

    # "Assemble" the supersystem for the displaced fragments and get integrals
    coeffs_grads = pickle.load(open("coeffs_grads.pkl", mode="rb"))
    state_coeffs = coeffs_grads[0]
    gradient_states = coeffs_grads[1]
    BeN = []
    dens = []
    dens_builder_stuff = []
    for m in range(int(n_frag)):
        state_obj, dens_var_1, dens_var_2, n_threads, Be = get_fci_states(displacement)
        for elem,coords in Be.atoms:  coords[2] += m * displacement    # displace along z
        for chg in monomer_charges[m]:
            state_obj[chg].coeffs = state_coeffs[m][chg]
        BeN.append(Be)
        dens.append(densities.build_tensors(state_obj, dens_var_1, dens_var_2, n_threads=n_threads))
        dens_builder_stuff.append([state_obj, dens_var_1, dens_var_2])

    ints = get_ints(BeN, project_core)

    def orthogonalize(U, eps=1e-15):  # with the transpose commented out, it orthogonalizes rows instead of columns
        n = len(U)
        V = U#.T
        for i in range(n):
            prev_basis = V[0:i]     # orthonormal basis before V[i]
            coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
            # subtract projections of V[i] onto already determined basis V[0:i]
            V[i] -= np.dot(coeff_vec, prev_basis).T
            if np.linalg.norm(V[i]) < eps:
                V[i][V[i] < eps] = 0.   # set the small entries to 0
            else:
                V[i] /= np.linalg.norm(V[i])
        return V#.T

    gs_energy, gs_state = get_xr_states(ints, dens, 0)

    for chg in monomer_charges[0]:
        #tmp = np.array(state_coeffs[0][chg]) - 2 * gradient_states[chg]  # factor of 2 is still from gradients (no empirical parameter for better convergence or similar stuff)
        tmp = np.array(state_coeffs[0][chg])
        tmp[-1] -= 2 * gradient_states[chg][-1]
        dens_builder_stuff[0][0][chg].coeffs = [i for i in orthogonalize(tmp)]
        #dens_builder_stuff[1][0][chg].coeffs = state_coeffs[1][chg]

    dens[0] = densities.build_tensors(*dens_builder_stuff[0], n_threads=n_threads, bra_det=False)
    #dens[1] = densities.build_tensors(*dens_builder_stuff[1], n_threads=n_threads, bra_det=False)      
    gs_energy_new, gs_state_new = get_xr_states(ints, dens, 0)

    print(gs_energy, gs_energy_new)







optimize_states(4.5, 1)
