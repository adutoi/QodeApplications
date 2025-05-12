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

# python [-u] dump-detdens.py 

import pickle
from lala import lala

if __name__=="__main__":

    basis = "6-31G", 9    # 9 spatial orbitals per 6-31G Be atom

    rho = lala(basis)

    pickle.dump(rho, open("rho/Be631g_detdens.pkl", "wb"))    # users responsibility to softlink rho/ to different volume if desired
