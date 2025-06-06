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
import tensorly
import multiprocessing
from qode.util           import sort_eigen
from qode.util.PyC       import Double
from qode.math.tensornet import tl_tensor, tensor_sum, raw
from qode.many_body.fermion_field import field_op
import compress

# states[n].coeffs  = [numpy.array, numpy.array, . . .]   One (effectively 1D) array of coefficients per n-electron state
# states[n].configs = [int, int, . . . ]                  Each int represents a configuration (has the same length as arrays in list above)



def _token_parser(options):
    parsed_options = {}
    if options is None:  options = []
    for option in options:
        if "=" in option:
            key, value = option.split("=")
            values = value.split(",")
            for i in range(len(values)):
                if values[i]=="True":
                    values[i] = True
                elif values[i]=="False":
                    values[i] = False
                else:
                    try:
                        temp = int(values[i])
                    except:
                        try:
                            temp = float(values[i])
                        except:
                            pass
                        else:
                            values[i] = temp
                    else:
                        values[i] = temp
            if len(values)==1 and (values[0] is True or values[0] is False):
                parsed_options[key] = values[0]
            else:
                parsed_options[key] = tuple(values)    # tuple bc equality might be checked
        else:
            parsed_options[option] = True
    def value(option):
        answer = None
        if option in parsed_options:
            answer = parsed_options[option]
        return answer
    return value

def _tens_wrap(tensor):
    return tl_tensor(tensorly.tensor(tensor, dtype=Double.tensorly))

def _vec(i, length):
    v = numpy.zeros((length,), dtype=Double.numpy, order="C")
    v[i] = 1
    return _tens_wrap(v)

def _compress(args):
    rho_ij, op_string, bra_chg, ket_chg, i, j, n_bras, n_kets, compress_args, natural_orbs, antisymm_abstract = args
    return i, n_bras, j, n_kets, compress.compress(rho_ij, op_string, bra_chg, ket_chg, i, j, compress_args, natural_orbs, antisymm_abstract, _tens_wrap)



# private version so that we can use "with pool" on the outside
def _build_tensors(states, n_orbs, n_elec_0, thresh, options, n_threads, pool):
    op_strings = {2:["aa", "caaa"], 1:["a", "caa", "ccaaa"], 0:["ca", "ccaa"]}
    densities = {}
    conj_densities = {}

    options = _token_parser(options)
    use_natural_orbs   = options("nat-orbs") is True                    # do compression in natural orbital rep? (default: no)
    antisymm_abstract  = options("abs-anti") is True                    # antisymmetry abstract in final rep, which might be original? (default: no)
    antisymm_numerical = (not antisymm_abstract) or use_natural_orbs    # numerically antisymmetrize in original rep? (default: yes)
    compress_args = options("compress")

    print("Computing densities ...")

    for bra_chg in states:
        bra_coeffs  = states[bra_chg].coeffs
        bra_configs = field_op.packed_configs(states[bra_chg].configs)
        for ket_chg in states:
            ket_coeffs  = states[ket_chg].coeffs
            ket_configs = field_op.packed_configs(states[ket_chg].configs)
            chg_diff = bra_chg - ket_chg
            if chg_diff in op_strings:
                for op_string in op_strings[chg_diff]:
                    if op_string not in densities:  densities[op_string] = {}
                    print(bra_chg, ket_chg, op_string)
                    # bit of a waste here ... computes i<j and i>j for chg_diff=0
                    rho = field_op.build_densities(op_string, n_orbs, bra_coeffs, ket_coeffs, bra_configs, ket_configs, thresh, wisdom=None, antisymmetrize=antisymm_numerical, n_threads=n_threads)
                    densities[op_string][bra_chg,ket_chg] = [[_tens_wrap(rho_ij) for rho_ij in rho_i] for rho_i in rho]

    print("Postprocessing ...")

    natural_orbs = None
    if use_natural_orbs:
        natural_orbs = {}
        for chg,_ in densities["ca"]:
            rho = densities["ca"][chg,chg]              # bra/ket charges must be the same for this string ...
            natural_orbs_chg = []
            for i in range(len(rho)):                   # ... which means the number of bras and kets are the same
                rho_ii = numpy.array(raw(rho[i][i]), dtype=Double.numpy, order="C")
                #print(chg, i, "deviation from symmetric:", numpy.linalg.norm(rho_ii - rho_ii.T))
                evals, evecs = sort_eigen(numpy.linalg.eigh(rho_ii), order="descending")
                natural_orbs_chg += [_tens_wrap(evecs)]
            natural_orbs[chg] = natural_orbs_chg

    for op_string in densities:
        for bra_chg,ket_chg in densities[op_string]:
            print("<>", op_string, bra_chg, ket_chg)
            rho = densities[op_string][bra_chg,ket_chg]
            temp_ij = tensor_sum()
            temp_ji = tensor_sum()
            #
            arguments = []
            n_bras = len(rho)
            for i,rho_i in enumerate(rho):
                n_kets = len(rho_i)
                for j,rho_ij in enumerate(rho_i):
                    if bra_chg!=ket_chg or i>=j:
                        #rho_ij = compress.compress(rho_ij, op_string, bra_chg, ket_chg, i, j, compress_args, natural_orbs, antisymm_abstract, _tens_wrap)
                        arguments += [(rho_ij, op_string, bra_chg, ket_chg, i, j, n_bras, n_kets, compress_args, natural_orbs, antisymm_abstract)]
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

    print("Writing to hard drive ...")
    return densities

def build_tensors(states, n_orbs, n_elec_0, thresh=1e-10, options=None, n_threads=1):
    if n_threads>1:
        with multiprocessing.Pool(n_threads, maxtasksperchild=1) as pool:    # to avoid errors on exit, both maxtasksperchild=1 ...
            densities = _build_tensors(states, n_orbs, n_elec_0, thresh, options, n_threads, pool)
            pool.close()                                                     # ... and pool.close() seem to be necessary
    else:
        densities = _build_tensors(states, n_orbs, n_elec_0, thresh, options, n_threads, None)
    return densities
