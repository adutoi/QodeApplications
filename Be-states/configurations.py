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



def all_nelec_configs(num_orb, num_elec, _outer_config=0):
    configs = []
    for p in range(num_elec-1, num_orb):
        config = _outer_config + 2**p
        if num_elec==1:  configs += [config]
        else:            configs += _all_nelec_configs(p, num_elec-1, _outer_config=config)
    return configs



def fci_configs(num_spat_orb, num_elec_dn, num_elec_up, num_core_orb):
    active_orb  = num_spat_orb - num_core_orb
    active_dn   = num_elec_dn  - num_core_orb
    active_up   = num_elec_up  - num_core_orb
    configs_dn  = all_nelec_configs(active_orb, active_dn)
    configs_up  = all_nelec_configs(active_orb, active_up)
    core_shift  = 2**num_core_orb
    core_config = core_shift - 1
    spin_shift  = 2**num_spat_orb
    for i in range(len(configs_dn)):
        configs_dn[i] *= core_shift
        configs_dn[i] += core_config
        configs_dn[i] *= spin_shift
    for i in range(len(configs_up)):
        configs_up[i] *= core_shift
        configs_up[i] += core_config
    configs = []
    for config_dn in configs_dn:
        for config_up in configs_up:
            configs += [config_dn + config_up]
    return configs



# The list of configs is interpreted as belonging to multiple systems, divided according to
# sysA_low_orbs, which gives the lowest-indexed orbital of each system (except for the last
# one which is assumed to be zero.  It then creates mutliply nested lists, where each state
# of a former system is associated with lists of all states of the the latter ones.  This
# should work with any number of systems, but it has only been tested for two.
def decompose_configs(configs, sysA_low_orbs):
    decomposed = []
    denom = 2**sysA_low_orbs[0]
    sysA_low_orbs = sysA_low_orbs[1:]
    sysA_prev = None
    for config in configs:
        sysA = config // denom
        low  = config %  denom
        if sysA!=sysA_prev:
            if sysA_prev is not None:
                if len(sysA_low_orbs)>0:  decomposed += [(sysA_prev, decompose_configs(current_sysB, sysA_low_orbs))]
                else:                     decomposed += [(sysA_prev, current_sysB)]
            current_sysB = []
            sysA_prev = sysA
        current_sysB += [low]
    if len(sysA_low_orbs)>0:  decomposed += [(sysA_prev, decompose_configs(current_sysB, sysA_low_orbs))]
    else:                     decomposed += [(sysA_prev, current_sysB)]
    return decomposed
