#    (C) Copyright 2023 Anthony D. Dutoi
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
from qode.util.PyC import BigInt, Double
from build_density_tensors import build_density_tensors

class empty(object):  pass



# Unused
#def configint_to_array(configint):
#    binary_str = "{:b}".format(configint)
#    return [p for p,occ in enumerate(reversed(binary_str)) if occ=='1']

def array_to_configint(array):
    configint = 0
    for p in array:  configint += 2**p
    return configint

def all_configs(active, n_elec, _beg=0, _config=None):
    if _config==None:  _config = []
    if n_elec==0:
        return [_config]
    else:
        new_configs = []
        for p in range(_beg, len(active)):
            new_config = _config + [active[p]]
            new_configs += all_configs(active, n_elec-1, _beg=p+1, _config=new_config)
        return new_configs

def add_core(core, configs):
    new_configs = []
    for config in configs:
        new_configs += [list(sorted(core+config))]
    return new_configs



def build_tensors(states, n_spatial_orb, spatial_core, n_threads=1):
    n_spatial_core = len(spatial_core)
    n_spin_orb     = 2 * n_spatial_orb
    n_spin_core    = 2 * n_spatial_core
    spin_core      = spatial_core + [p+n_spatial_orb for p in spatial_core]
    spin_active    = [p for p in range(n_spin_orb) if p not in spin_core]

    z_lists = {}
    for chg,data in states.items():
        n_states  = len(data.coeffs)
        n_configs = len(data.configs)
        n_elec    = data.configs[0].bit_count()
        z_lists[chg] = empty()
        z_lists[chg].coeffs  = [numpy.zeros(n_configs, dtype=Double.numpy) for _ in range(n_states)]
        z_lists[chg].configs = add_core(spin_core, all_configs(spin_active, n_elec-n_spin_core))
        if len(z_lists[chg].configs)!=n_configs:
            raise RuntimeError("assumptions about configuration spaces must be faulty") 
        for a,config in enumerate(z_lists[chg].configs):
            i = data.configs.index(array_to_configint(config))    # list.index() will raise an exception if not found
            for z in range(n_states):
                z_lists[chg].coeffs[z][a] = data.coeffs[z][i]
        z_lists[chg].coeffs  = numpy.array(z_lists[chg].coeffs,  dtype=Double.numpy)
        z_lists[chg].configs = numpy.array(z_lists[chg].configs, dtype=BigInt.numpy)

    return build_density_tensors(z_lists, n_spatial_orb, n_spatial_core, n_threads)
