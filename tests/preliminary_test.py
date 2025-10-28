#    (C) Copyright 2025 Marco Bauer
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
import numpy as np
from mains.workflow import run_xr

# TODO: rename this file

def test_grad_free_opt():
    test = run_xr(4.5, 0, 1, single_thresh=1/6, double_thresh=1/4, triple_thresh=1/2.5,
                  grad_level="herm", state_prep=True, target_state=[0, 1], dens_filter_thresh_solver=1e-6, backend="psi4 in_house",
                  load_ref="ref_data")
    ref = (np.array([-31.10730539+0.j, -31.04883404+0.j]), np.array([-31.1073116 +0.j, -31.04884124+0.j]))
    np.testing.assert_allclose(test, ref, atol=1e-7)  # rtol=1e-7 is the default

#def test_covariant_opt():
#    test = run_xr(4.5, 50, 1, single_thresh=1/6, double_thresh=1/4, triple_thresh=1/2.5,
#                  grad_level="full", state_prep=False, target_state=[0], dens_filter_thresh_solver=1e-6, backend="psi4 in_house",
#                  load_ref="ref_data")
#    ref = ([-31.10728315+0.j], np.array([-31.10731138+0.j, -31.04013036+0.j]))
#    np.testing.assert_allclose(test[0], ref[0], atol=1e-7)  # rtol=1e-7 is the default
#    np.testing.assert_allclose(test[1], ref[1], atol=1e-7)  # rtol=1e-7 is the default

def test_contravariant_opt():
    test = run_xr(4.5, 50, 1, single_thresh=1/6, double_thresh=1/4, triple_thresh=1/2.5,
                  grad_level="herm", state_prep=False, target_state=[0], dens_filter_thresh_solver=1e-6, backend="psi4 in_house",
                  load_ref="ref_data")
    ref = (np.array(-31.10728521+0.j), np.array([-31.10731553+0.j, -31.04007645+0.j]))
    np.testing.assert_allclose(test[0], ref[0], atol=1e-7)  # rtol=1e-7 is the default
    np.testing.assert_allclose(test[1], ref[1], atol=1e-7)  # rtol=1e-7 is the default

if __name__ == "__main__":
    test_grad_free_opt()
    test_contravariant_opt()
