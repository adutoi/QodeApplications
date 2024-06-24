#    (C) Copyright 2023, 2024 Anthony D. Dutoi
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

# python [-u] main-631g.py <nthreads> <distance> thresh|nstates=<value> [compress=<val,val,...>] [nat-orbs] [abs-anti]

import sys
import pickle
from build_Be_rho import build_Be_rho

def _abbrev(token):
    tokens = token.split("=")
    return tokens[1] if len(tokens)==2 else tokens[0]

if __name__=="__main__":

    basis        = "6-31G", 9    # 9 spatial orbitals per 6-31G Be atom
    n_threads    =   int(sys.argv[1])
    dist         = float(sys.argv[2])
    statesthresh =       sys.argv[3]
    options      =       sys.argv[4:]

    label = "_".join(_abbrev(arg) for arg in sys.argv[2:])
    rho = build_Be_rho(basis, dist, statesthresh, options, n_threads)

    pickle.dump(rho, open("rho/Be631g_{}.pkl".format(label), "wb"))    # users responsibility to softlink rho/ to different volume if desired
