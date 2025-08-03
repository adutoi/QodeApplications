import pickle
import numpy as np

class empty(object):  pass    # sorry about the need for this. will be deprecated soon.

D = pickle.load(open("../../Downloads/Be631g-CIS.pkl","rb"))  # this is cis with densities with non-converged lin combs of hf states as basis, and densities from FCI procedure

#print(D.rho["ca"][0,0][0][0])

#print(D.__dict__.keys())

#print(D.atoms, D.n_elec_ref, D.basis, D.state_indices)
#print(D.basis.Mocoeffs)


import sys
#import numpy
import torch
import tensorly as tl
import qode.util
import qode.math
import excitonic
import diagrammatic_expansion   # defines information structure for housing results of diagram evaluations
import XR_term                  # knows how to use ^this information to pack a matrix for use in XR model
import S_diagrams               # contains definitions of actual diagrams needed for S operator in BO rep
import Sn_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import St_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import Su_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
import Sv_diagrams              # contains definitions of actual diagrams needed for SH operator in BO rep
from   get_ints import get_ints
from   Be631g   import monomer_data as Be
from tendot import tendot
#import ray
import time

torch.set_num_threads(4)
tl.set_backend("pytorch")


# Information about the Be2 supersystem
n_frag       = 2
#displacement = sys.argv[1]
states       = "load=states:16-115-550:thresh=1e-6:4.5:u.pickle/4.5"
n_states     = ("all","all","all","all","all")

displacement = 4.5  # this is just a dummy, because the function requires the variable, even though it is not necessary in the following test

# "Assemble" the supersystem from the displaced fragments
BeN = [Be((0,0,m*float(displacement))) for m in range(int(n_frag))]

# Load states and get integrals
#for m,frag in enumerate(BeN):  frag.load_states(states, n_states)      # load the density tensors
BeN[0].load_states(states, n_states)
#symm_ints, bior_ints, nuc_rep = get_ints(BeN)

#print(BeN[0].rho["ca"][0,0][0][0])



# function to stupid adcc ordering to reorder spin block ordering
def reorder_to_spin_block_ca(ten, n_occ_a=2, n_occ_b=2, n_virt_a=7, n_virt_b=7):
    tot = n_occ_a + n_occ_b + n_virt_a + n_virt_b
    if (tot, tot) != ten.shape:
        raise ValueError("reorder function orbital spaces don't fit tensor shape")
    # define some shortcuts first
    n_occ_tot = n_occ_a + n_occ_b
    occ_and_virt_a = n_occ_tot + n_virt_a
    occ_a_line = np.hstack((ten[:n_occ_a, :n_occ_a], ten[:n_occ_a, n_occ_tot:occ_and_virt_a],
                                ten[:n_occ_a, n_occ_b:n_occ_tot], ten[:n_occ_a, occ_and_virt_a:]))
    virt_a_line = np.hstack((ten[n_occ_tot:occ_and_virt_a, :n_occ_a], ten[n_occ_tot:occ_and_virt_a, n_occ_tot:occ_and_virt_a],
                                 ten[n_occ_tot:occ_and_virt_a, n_occ_a:n_occ_tot], ten[n_occ_tot:occ_and_virt_a, occ_and_virt_a:]))
    occ_b_line = np.hstack((ten[n_occ_a:n_occ_tot, :n_occ_a], ten[n_occ_a:n_occ_tot, n_occ_tot:occ_and_virt_a],
                                ten[n_occ_a:n_occ_tot, n_occ_a:n_occ_tot], ten[n_occ_a:n_occ_tot, occ_and_virt_a:]))
    virt_b_line = np.hstack((ten[occ_and_virt_a:, :n_occ_a], ten[occ_and_virt_a:, n_occ_tot:occ_and_virt_a],
                                 ten[occ_and_virt_a:, n_occ_a:n_occ_tot], ten[occ_and_virt_a:, occ_and_virt_a:]))
    return np.vstack((occ_a_line, virt_a_line, occ_b_line, virt_b_line))



# function to stupid adcc ordering to reorder spin block ordering
def reorder_to_spin_block_ccaa(ten, n_occ_a=2, n_occ_b=2, n_virt_a=7, n_virt_b=7):
    tot = n_occ_a + n_occ_b + n_virt_a + n_virt_b
    #if (tot, tot) != ten.shape:
    if (tot, tot, tot, tot) != ten.shape:
        print(ten.shape)
        raise ValueError("reorder function orbital spaces don't fit tensor shape")
    # define some shortcuts first
    n_occ_tot = n_occ_a + n_occ_b
    occ_and_virt_a = n_occ_tot + n_virt_a
    ret = np.concatenate((ten[:n_occ_a, :, :, :], ten[n_occ_tot:occ_and_virt_a, :, :, :], ten[n_occ_a:n_occ_tot, :, :, :], ten[occ_and_virt_a:, :, :, :]), axis=0)
    ret = np.concatenate((ret[:, :n_occ_a, :, :], ret[:, n_occ_tot:occ_and_virt_a, :, :], ret[:, n_occ_a:n_occ_tot, :, :], ret[:, occ_and_virt_a:, :, :]), axis=1)
    ret = np.concatenate((ret[:, :, :n_occ_a, :], ret[:, :, n_occ_tot:occ_and_virt_a, :], ret[:, :, n_occ_a:n_occ_tot, :], ret[:, :, occ_and_virt_a:, :]), axis=2)
    ret = np.concatenate((ret[:, :, :, :n_occ_a], ret[:, :, :, n_occ_tot:occ_and_virt_a], ret[:, :, :, n_occ_a:n_occ_tot], ret[:, :, :, occ_and_virt_a:]), axis=3)
    return ret




adcc_to_cis_map = {}  # sorting the states

for i in range(len(D.rho["ca"][0,0])):
    D_adc = reorder_to_spin_block_ca(np.asarray(BeN[0].rho["ca"][0,0][i][i]))
    for j in range(len(D.rho["ca"][0,0])):
        D_cis = D.rho["ca"][0,0][j][j]
        if np.linalg.norm(D_cis - D_adc) <= 1e-15:
            #print(f"adcc state {i} maps to cis state {j}")
            adcc_to_cis_map[i] = j

print(adcc_to_cis_map)

#print(np.asarray(BeN[0].rho["ca"][0,0][0][0]))
#print(reorder_to_spin_block_ca(np.asarray(BeN[0].rho["ca"][0,0][0][0])))


ca_diff_norms = {}

for i in range(len(D.rho["ca"][0,0])):
    for j in range(len(D.rho["ca"][0,0][i])):
        D_adc = reorder_to_spin_block_ca(np.asarray(BeN[0].rho["ca"][0,0][i][j]))
        #if (i >= 8 and j < 8) or (i < 8 and j >= 8):  # factor of -1 for certain states...no problem, just accounting for global phase
        #    D_adc *= -1
        factor = 1
        if i >= 8:
            factor *= -1
        if j >= 8:
            factor *= -1
        D_adc *= factor
        diff = D_adc - D.rho["ca"][0,0][adcc_to_cis_map[i]][adcc_to_cis_map[j]]
        #print((i, j), np.linalg.norm(diff))
        ca_diff_norms[(i, j)] = np.linalg.norm(diff)

print(f"sum of ca_diff_norms divided by n_elements {np.sum(list(ca_diff_norms.values())) / len(list(ca_diff_norms.values()))}")

#print(reorder_to_spin_block(np.asarray(BeN[0].rho["ca"][0,0][8][8]))[:9, :9])
#print(D.rho["ca"][0,0][adcc_to_cis_map[8]][adcc_to_cis_map[8]][:9, :9])
#print(reorder_to_spin_block(np.asarray(BeN[0].rho["ca"][0,0][8][9]))[:9, :9])
#print(D.rho["ca"][0,0][adcc_to_cis_map[8]][adcc_to_cis_map[9]][:9, :9])

ccaa_diff_norms = {}

for i in range(len(D.rho["ccaa"][0,0])):
    for j in range(len(D.rho["ccaa"][0,0][i])):
        factor = 1
        if i >= 8:
            factor *= -1
        if j >= 8:
            factor *= -1
        D_adc = reorder_to_spin_block_ccaa(np.asarray(BeN[0].rho["ccaa"][0,0][i][j]))
        D_adc *= factor
        diff = D_adc - D.rho["ccaa"][0,0][adcc_to_cis_map[i]][adcc_to_cis_map[j]]
        ccaa_diff_norms[(i, j)] = np.linalg.norm(diff)

print(f"sum of ccaa_diff_norms divided by n_elements {np.sum(list(ccaa_diff_norms.values())) / len(list(ccaa_diff_norms.values()))}")

print("largest error norm in ccaa ", np.max(list(ccaa_diff_norms.values())))





#print(D.rho["ccaa"][0,0][0][0][:2, :2, :2, :2])



