#    (C) Copyright 2025 Anthony D. Dutoi
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
import pickle
import numpy
from mains.xr_ccsd import main

def testP():
    frags = [pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb")),
             pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb"))]
    E = main(order="proper", frags=frags, displacement=4.5, project_core=False)
    numpy.testing.assert_allclose(E, -29.225774484997, atol=1e-10)

def testM():
    frags = [pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb")),
             pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb"))]
    E = main(order="M=1", frags=frags, displacement=4.5, project_core=False)
    numpy.testing.assert_allclose(E, -27.343955072182, atol=1e-10)

def test0():
    frags = [pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb")),
             pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb"))]
    E = main(order=0, frags=frags, displacement=4.5, project_core=False)
    numpy.testing.assert_allclose(E, -29.225774484997, atol=1e-10)

def test1():
    frags = [pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb")),
             pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb"))]
    E = main(order=1, frags=frags, displacement=4.5, project_core=False)
    numpy.testing.assert_allclose(E, -29.225805660732, atol=1e-10)

def test2():
    frags = [pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb")),
             pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb"))]
    E = main(order=2, frags=frags, displacement=4.5, project_core=False)
    numpy.testing.assert_allclose(E, -29.225801454382, atol=1e-10)

def test3():
    frags = [pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb")),
             pickle.load(open(f"../hermitian-XRCC/rho/Be-Be_0_6-31G_nth_compress-factored.pkl", "rb"))]
    E = main(order=3, frags=frags, displacement=4.5, project_core=False)
    numpy.testing.assert_allclose(E, -29.225801249098, atol=1e-10)

if __name__ == "__main__":
    testP()
    testM()
    test0()
    test1()
    test2()
    test3()
