#    (C) Copyright 2023 Anthony Dutoi
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


def _all_configs(active_orbs, num_active_elec, static_config):
    configs = []
    for p in range(num_active_elec-1, len(active_orbs)):
        config = static_config + 2**active_orbs[p]
        if num_active_elec==1:  configs += [config]
        else:                   configs += _all_configs(active_orbs[:p], num_active_elec-1, config)
    return configs

def all_configs(num_tot_orb, num_active_elec, frozen_occ_orbs=None, frozen_vrt_orbs=None):
    if frozen_occ_orbs is None:  frozen_occ_orbs = []
    if frozen_vrt_orbs is None:  frozen_vrt_orbs = []
    frozen_occ_orbs = set(frozen_occ_orbs)
    frozen_vrt_orbs = set(frozen_vrt_orbs)
    static_config = 0
    active_orbs = []
    for p in range(num_tot_orb):
        if p in frozen_occ_orbs:
            static_config += 2**p
        elif p not in frozen_vrt_orbs:
            active_orbs += [p]
    return _all_configs(active_orbs, num_active_elec, static_config)

# The list of configs is interpreted as belonging to multiple systems, divided according to
# sysA_low_orbs, which gives the lowest-indexed orbital of each system (except for the last
# one which is assumed to be zero.  It then creates mutliply nested lists, where each state
# of a former system is associated with lists of all states of the the latter ones.  This
# should work with any number of systems, but it has only been tested for two.
def decompose_configs(configs, orb_counts):
    nested = []
    orb_counts = orb_counts[:-1]
    shift = 2**sum(orb_counts)
    configA_prev = None
    for config in configs:
        configA  = config // shift
        configBZ = config %  shift
        if configA!=configA_prev:
            if configA_prev is not None:
                if len(orb_counts)>1:  nested += [(configA_prev, decompose_configs(configsBZ, orb_counts))]
                else:                  nested += [(configA_prev, configsBZ)]
            configsBZ = []
            configA_prev = configA
        configsBZ += [configBZ]
    if len(orb_counts)>1:  nested += [(configA_prev, decompose_configs(configsBZ, orb_counts))]
    else:                  nested += [(configA_prev, configsBZ)]
    return nested

def config_combination(orb_counts):
    shifts = [1]
    orb_count_tot = 0
    for orb_count in orb_counts[:-1]:
        orb_count_tot += orb_count
        shifts += [2**orb_count_tot]
    def combine_configs(*configsX):
        config = 0
        for configX,shift in zip(reversed(configsX),shifts):
            config += configX*shift
        return config
    return combine_configs

def combine_decomposed(nested, orb_counts):
    orb_count  = orb_counts[-1]
    orb_counts = orb_counts[:-1]
    combine_configs = config_combination([sum(orb_counts), orb_count])
    configs = []
    for configA,nestedBZ in nested:
        if len(orb_counts)>1:  configsBZ = combine_decomposed(nestedBZ, orb_counts)
        else:                  configsBZ = nestedBZ
        for configBZ in configsBZ:
            configs += [combine_configs(configA, configBZ)]
    return configs

def _tensor_pdt_nested(configsX):
    configsA = configsX[-1]
    configsX = configsX[:-1]
    if len(configsX)==0:
        nested = configsA
    else:
        nested = []
        for configA in configsA:
            nested += [(configA, _tensor_pdt_nested(configsX))]
    return nested

def tensor_product_configs(configsX, orb_counts):
    return combine_decomposed(_tensor_pdt_nested(configsX), orb_counts)



def print_configs(nested, orb_counts, _indent=""):
    num_orb = orb_counts[-1]
    orb_counts = orb_counts[:-1]
    formatter = "{{:0{}b}}".format(num_orb)
    if len(orb_counts)==0:
        for config in nested:
            print(_indent, formatter.format(config))
    else:
        for configA,nestedBZ in nested:
            print(_indent, formatter.format(configA))
            print_configs(nestedBZ, orb_counts, _indent+" "*num_orb)



if __name__ == "__main__":
    configs = _all_configs([0,2,4,6,8], 3, 0b0010101010)
    print_configs(configs, [10])
    print()
    configs = all_configs(10, 4, frozen_occ_orbs=[0,5], frozen_vrt_orbs=[1,6])
    print_configs(configs, [10])
    print()
    configs = all_configs(6, 3)
    print_configs(configs, [6])
    print()
    nested = decompose_configs(configs, [3,3])
    print_configs(nested, [3,3])
    print()
    configs = combine_decomposed(nested, [3,3])
    print_configs(configs, [6])
    print()
    configs = all_configs(4, 2)
    print_configs(configs, [4])
    print()
    nested = _tensor_pdt_nested([configs, configs])
    print_configs(nested, [4,4])
    print()
    configs = tensor_product_configs([configs, configs], [4,4])
    print_configs(configs, [8])
