#    (C) Copyright 2018, 2019, 2023 Anthony D. Dutoi and Yuhong Liu
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
#import multiprocessing
from qode.util           import sort_eigen
from qode.util.PyC       import Double
from qode.math.tensornet import tl_tensor, tensor_sum, raw
import field_op
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



def build_tensors(states, n_orbs, n_elec_0, thresh=1e-10, options=None, n_threads=1):
    #pool = multiprocessing.Pool(n_threads)

    #op_strings = {2:["aa", "caaa"], 1:["a", "caa", "ccaaa"], 0:["ca", "ccaa"]}
    op_strings = {2:["aa"], 1:["a", "caa"], 0:["ca", "ccaa"]}
    densities = {}
    conj_densities = {}

    options = _token_parser(options)
    use_natural_orbs   = options("nat-orbs") is True                    # do compression in natural orbital rep? (default: no)
    antisymm_abstract  = options("abs-anti") is True                    # antisymmetry abstract in final rep, which might be original? (default: no)
    antisymm_numerical = (not antisymm_abstract) or use_natural_orbs    # numerically antisymmetrize in original rep? (default: yes)
    compress_args = options("compress")
    if options("bra_det"):
        op_strings[-1] = ["c", "cca"]
        op_strings[-2] = ["cc"]

    print("Computing densities ...")

    for bra_chg in states:
        bra_configs = field_op.packed_configs(states[bra_chg].configs)
        if options("bra_det"):
            #print("bra det True")
            bra_coeffs = [i for i in numpy.eye(len(bra_configs))]
        else:
            bra_coeffs  = states[bra_chg].coeffs
        for ket_chg in states:
            ket_coeffs  = states[ket_chg].coeffs
            #if len(coeffs[1].keys()) != 0:
            #    ket_coeffs = coeffs[1][ket_chg]  # also let this allow for differing number of states in the future
            #print("len kets", len(ket_coeffs))
            #if len(bra_coeffs) > 200 and op_string == "ccaa":
            ket_configs = field_op.packed_configs(states[ket_chg].configs)
            chg_diff = bra_chg - ket_chg
            if chg_diff in op_strings:
                for op_string in op_strings[chg_diff]:
                    if op_string not in densities:  densities[op_string] = {}
                    print(bra_chg, ket_chg, op_string)
                    if (options("bra_det") and op_string == "ccaa") and (bra_chg == -1 and ket_chg == -1):
                        #print("bra det -1 -1 ccaa")
                        # only do this for ccaa -1 -1
                        tmp = _tens_wrap(numpy.ones(n_orbs) * 1e-10)  # choosing this as actual zeros leads to numerical inconsistencies in the gradients
                        print("this density was taken as an almost zero tensor. Beware, that this is an approximation!")
                        rho = [[tmp(0) @ tmp(1) @ tmp(2) @ tmp(3) for dummy in range(len(ket_coeffs))] for dummy2 in range(len(bra_coeffs))]
                        densities[op_string][bra_chg,ket_chg] = rho
                    else:
                        # bit of a waste here ... computes i<j and i>j for chg_diff=0
                        rho = field_op.build_densities(op_string, n_orbs, bra_coeffs, ket_coeffs, bra_configs, ket_configs, thresh, wisdom=None, antisymmetrize=antisymm_numerical, n_threads=n_threads)
                        densities[op_string][bra_chg,ket_chg] = [[_tens_wrap(rho_ij) for rho_ij in rho_i] for rho_i in rho]
                    #if options("bra_det"):
                    #    ret = tensor_sum()
                    #    indices = tuple(p+2 for p in range(len(op_string)))
                    #    ret += _vec(i,n_bras)(0) @ _vec(j,n_kets)(1) @ rho_ij(*indices)
                    #    ret += densities[op_string][bra_chg,ket_chg]

    print("Postprocessing ...")

    #if options("bra_det"):
    #    pass
    #else:
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

    #if not options("bra_det"):
    for op_string in densities:
        for bra_chg,ket_chg in densities[op_string]:
            print(op_string, bra_chg, ket_chg)
            rho = densities[op_string][bra_chg,ket_chg]
            #if op_string == "ccaa" and options("bra_det"):
            #    if bra_chg == -1 and ket_chg == -1:
            #        continue  # ccaa -1 to -1 is already in decomposed form, because it's so large
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
            #if n_bras > 200 and op_string == "ccaa":  # better: if type(rho_ij) == qode.math.tensornet.base.to_contract:
            if options("bra_det") or compress_args == None:# and bra_chg==ket_chg:  # better also decompose for equal charges, but with bras different from kets the accumulator needs to be populated differently
                values = [(args[4], args[6], args[5], args[7], args[0]) for args in arguments]
            else:
                values = [_compress(args) for args in arguments]
            #values = pool.map(_compress, arguments)
            for i, n_bras, j, n_kets, rho_ij in values:
                #if True:
                #    if True:
                indices = tuple(p+2 for p in range(len(op_string)))
                temp_ij += _vec(i,n_bras)(0) @ _vec(j,n_kets)(1) @ rho_ij(*indices)
                #if not options("bra_det"):
                rev_indices = tuple(reversed(indices))
                if bra_chg==ket_chg and not options("bra_det"):
                    if i!=j:# or options("bra_det"):  # if bra_det then bra and ket are not equal
                        #if options("bra_det"):
                        #    print(_vec(j,n_kets).shape, _vec(i,n_bras).shape, raw(rho_ij).shape)
                        #    print("inds:", 0, 1, *rev_indices)
                        temp_ij += _vec(j,n_kets)(0) @ _vec(i,n_bras)(1) @ rho_ij(*rev_indices)
                else:
                    temp_ji += _vec(j,n_kets)(0) @ _vec(i,n_bras)(1) @ rho_ij(*rev_indices)
            #
            densities[op_string][bra_chg,ket_chg] = temp_ij
            if bra_chg!=ket_chg and not options("bra_det"):  # bra_det densities are not symmetric!
                rev_op_string = op_string[::-1].replace("c","x").replace("a","c").replace("x","a")
                if rev_op_string not in conj_densities:  conj_densities[rev_op_string] = {}
                conj_densities[rev_op_string][ket_chg,bra_chg] = temp_ji

    for k,v in conj_densities.items():  densities[k] = v

    densities["n_elec"]    = {chg:(n_elec_0-chg)          for chg in states}
    densities["n_states"]  = {chg:len(states[chg].coeffs) for chg in states}
    if options("bra_det"):
        densities["n_states_bra"]  = {chg:len(states[chg].configs) for chg in states}
    else:
        densities["n_states_bra"]  = densities["n_states"]

    print("Writing to hard drive ...")
    return densities
