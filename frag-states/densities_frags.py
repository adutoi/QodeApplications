#    (C) Copyright 2018, 2019, 2023, 2024 Anthony D. Dutoi and Yuhong Liu
# 
#    This file is part of QodeApplications.
# 
#    QodeApplications is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    QodeApplications is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with QodeApplications.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy
import multiprocessing
from qode.util           import sort_eigen, indented
from qode.util.PyC       import Double
from qode.math.tensornet import evaluate
from qode.math.permute   import permutations_by_parity
from qode.many_body.fermion_field import field_op
import XR_tensor
import compress_frags

# states[n].coeffs  = [numpy.array, numpy.array, . . .]   One (effectively 1D) array of coefficients per n-electron state
# states[n].configs = [int, int, . . . ]                  Each int represents a configuration (has the same length as arrays in list above)



def _vec(i, length):
    v = numpy.zeros((length,), dtype=Double.numpy, order="C")
    v[i] = 1
    return XR_tensor.init(v)

def _compress(args):
    rho_ij, op_string, bra_chg, ket_chg, i, j, n_bras, n_kets, compress_args, natural_orbs, antisymm_abstract = args
    return i, n_bras, j, n_kets, compress_frags.compress(rho_ij, op_string, bra_chg, ket_chg, i, j, compress_args, natural_orbs, antisymm_abstract)

# The antisymmetrize flag to field_op.build_densities takes forever, so set it to False and run this instead.
def _antisymmetrize(antisymmetrize, tensor, op_string):
    if antisymmetrize:    # burying the "if" in here cleans up the calling code
        c_count = op_string.count("c")
        a_count = op_string.count("a")
        c_indices = list(range(0,       c_count        ))
        a_indices = list(range(c_count, c_count+a_count))
        c_perms = permutations_by_parity(c_indices)
        a_perms = permutations_by_parity(a_indices)
        result = XR_tensor.zeros()    # takes its shape from summed terms
        for perm in c_perms[0]:    # all the + permutations of c indices
            result += tensor(*(perm+a_indices))
        for perm in c_perms[1]:    # all the - permutations of c indices
            result -= tensor(*(perm+a_indices))
        tensor = evaluate(result)      # this "layering/nesting" of the c and a permutations is ...
        result = XR_tensor.zeros()     # ... more efficient than doing one big sum for both at the same time
        for perm in a_perms[0]:    # all the + permutations of a indices
            result += tensor(*(c_indices+perm))
        for perm in a_perms[1]:    # all the - permutations of a indices
            result -= tensor(*(c_indices+perm))
        tensor = evaluate(result)
    return tensor



# private version so that we can use "with pool" on the outside
def _build_tensors(states, n_orbs, n_elec_0, op_strings, thresh, options, printout, n_threads, pool):
    densities = {}
    conj_densities = {}

    use_natural_orbs   = options.nat_orbs                    # do compression in natural orbital rep? (default: no)
    antisymm_abstract  = options.abs_anti                    # antisymmetry abstract in final rep, which might be original? (default: no)
    antisymm_numerical = (not antisymm_abstract) or use_natural_orbs    # numerically antisymmetrize in original rep? (default: yes)

    printdent = indented(printout)

    printout("Computing densities ...")

    for bra_chg in states:
        bra_coeffs  = states[bra_chg].coeffs
        bra_configs = field_op.packed_configs(states[bra_chg].configs)
        for ket_chg in states:
            printdent(bra_chg, ket_chg)
            ket_coeffs  = states[ket_chg].coeffs
            ket_configs = field_op.packed_configs(states[ket_chg].configs)
            chg_diff = bra_chg - ket_chg
            if chg_diff in op_strings:
                for op_string in op_strings[chg_diff]:
                    if op_string not in densities:  densities[op_string] = {}
                    #printout("  ", bra_chg, ket_chg, op_string)
                    # bit of a waste here ... computes i<j and i>j for chg_diff=0
                    rho = field_op.build_densities(op_string, n_orbs, bra_coeffs, ket_coeffs, bra_configs, ket_configs, thresh, wisdom=None, antisymmetrize=False, printout=indented(printdent), n_threads=n_threads)
                    densities[op_string][bra_chg,ket_chg] = [[_antisymmetrize(antisymm_numerical, XR_tensor.init(rho_ij), op_string) for rho_ij in rho_i] for rho_i in rho]

    printout("Postprocessing/compressing ...")

    natural_orbs = None
    if use_natural_orbs:
        natural_orbs = {}
        for chg,_ in densities["ca"]:
            rho = densities["ca"][chg,chg]              # bra/ket charges must be the same for this string ...
            natural_orbs_chg = []
            for i in range(len(rho)):                   # ... which means the number of bras and kets are the same
                rho_ii = numpy.array(XR_tensor.raw(rho[i][i]), dtype=Double.numpy, order="C")
                #printout(chg, i, "deviation from symmetric:", numpy.linalg.norm(rho_ii - rho_ii.T))
                evals, evecs = sort_eigen(numpy.linalg.eigh(rho_ii), order="descending")
                natural_orbs_chg += [XR_tensor.init(evecs)]
            natural_orbs[chg] = natural_orbs_chg

    for op_string in densities:
        for bra_chg,ket_chg in densities[op_string]:
            printdent(op_string, bra_chg, ket_chg)
            rho = densities[op_string][bra_chg,ket_chg]
            temp_ij = XR_tensor.zeros()    # takes its shape from summed terms
            temp_ji = XR_tensor.zeros()    # takes its shape from summed terms
            #
            arguments = []
            n_bras = len(rho)
            for i,rho_i in enumerate(rho):
                n_kets = len(rho_i)
                for j,rho_ij in enumerate(rho_i):
                    if bra_chg!=ket_chg or i>=j:
                        #rho_ij = compress_frags.compress(rho_ij, op_string, bra_chg, ket_chg, i, j, options.compress, natural_orbs, antisymm_abstract)
                        arguments += [(rho_ij, op_string, bra_chg, ket_chg, i, j, n_bras, n_kets, options.compress, natural_orbs, antisymm_abstract)]
            if pool is None:
                values = [_compress(args) for args in arguments]
            else:
                values = pool.map(_compress, arguments)    # instead of pool, make pool.map the function argument, replaceable with map
            for i, n_bras, j, n_kets, rho_ij in values:
                if True:        # if these two lines and the three above are deleted, and the line above that is uncommented, ...
                    if True:    # ... then even the indentation will be right to obtain working code
                        indices = tuple(p+2 for p in range(len(op_string)))
                        temp_ij += _vec(i,n_bras)(0) @ _vec(j,n_kets)(1) @ rho_ij(*indices)
                        rev_indices = tuple(reversed(indices))
                        if bra_chg==ket_chg:
                            if i!=j:
                                temp_ij += _vec(j,n_kets)(0) @ _vec(i,n_bras)(1) @ rho_ij(*rev_indices)
                        else:
                            temp_ji += _vec(j,n_kets)(0) @ _vec(i,n_bras)(1) @ rho_ij(*rev_indices)
            #
            densities[op_string][bra_chg,ket_chg] = temp_ij
            if bra_chg!=ket_chg:
                rev_op_string = op_string[::-1].replace("c","x").replace("a","c").replace("x","a")
                if rev_op_string not in conj_densities:  conj_densities[rev_op_string] = {}
                conj_densities[rev_op_string][ket_chg,bra_chg] = temp_ji

    for k,v in conj_densities.items():  densities[k] = v

    densities["n_elec"]   = {chg:(n_elec_0-chg)          for chg in states}
    densities["n_states"] = {chg:len(states[chg].coeffs) for chg in states}

    printout("Done building densities ...")
    return densities

def build_tensors(frags, op_strings, thresh=1e-10, options=None, printout=print, n_threads=1):
    for m,frag in enumerate(frags):
        printout(f"Fragment {m}")
        states, n_orbs, n_elec_0 = frag.states, 2*frag.basis.n_spatial_orb, frag.n_elec_ref
        if n_threads>1:
            with multiprocessing.Pool(n_threads, maxtasksperchild=1) as pool:    # to avoid errors on exit, both maxtasksperchild=1 ...
                densities = _build_tensors(states, n_orbs, n_elec_0, op_strings, thresh, options, indented(printout), n_threads, pool)
                pool.close()                                                     # ... and pool.close() seem to be necessary
        else:
            densities = _build_tensors(states, n_orbs, n_elec_0, op_strings, thresh, options, indented(printout), n_threads, None)
        frag.rho = densities
    return "compress"
