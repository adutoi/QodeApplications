from mains.workflow import run_xr
import numpy as np

# make sure you set the following PYTHONPATHs
#export PYTHONPATH=$PYTHONPATH:/home/usr/Qode 
#export PYTHONPATH=$PYTHONPATH:/home/usr/QodeApplications/hermitian-XRCC
#export PYTHONPATH=$PYTHONPATH:/home/usr/QodeApplications/Be-states
#export PYTHONPATH=$PYTHONPATH:/home/usr/QodeApplications/StateSpaceOptimizer

# it is recommended to generate your own testdata before applying changes
# in order to do so, read the instructions at the bottom of the workflow.py file

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

