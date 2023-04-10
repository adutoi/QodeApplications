/*   (C) Copyright 2018, 2019 Anthony D. Dutoi and Yuhong Liu
 *
 *   This file is part of Qode.
 *
 *   Qode is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Qode is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Qode.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "PyC_types.h"



PyFloat monomer(PyInt n_orb, Double* Rca, Double* Rccaa, Double* h, Double* V)
	{
	PyFloat H = 0;
	for (PyInt p=0;  p<n_orb;  p++)
		{
		for (PyInt q=0;  q<n_orb;  q++)
			{
			for (PyInt r=0;  r<n_orb;  r++)
				{
				for (PyInt s=0;  s<n_orb;  s++)
					{
					H += V[((p*n_orb + q)*n_orb + r)*n_orb + s] * Rccaa[((p*n_orb + q)*n_orb + s)*n_orb + r];
					}
				}
			}
		}
	for (PyInt p=0;  p<n_orb;  p++)
		{
		for (PyInt q=0;  q<n_orb;  q++)
			{
			H += h[p*n_orb + q] * Rca[p*n_orb + q];
			}
		}
	return H;
	}



PyFloat dimer_2min2pls(PyInt n_orb1, PyInt n_orb2, Double* Rcc1, Double* Raa2, Double* V)
	{
	PyFloat H = 0;
	for (PyInt p1=0;  p1<n_orb1;  p1++)
		{
		for (PyInt q1=0;  q1<n_orb1;  q1++)
			{
			for (PyInt r2=0;  r2<n_orb2;  r2++)
				{
				for (PyInt s2=0;  s2<n_orb2;  s2++)
					{
					H += V[((p1*n_orb1 + q1)*n_orb2 + r2)*n_orb2 + s2] * Rcc1[p1*n_orb1 + q1] * Raa2[s2*n_orb2 + r2];
					}
				}
			}
		}
	return H;
	}



PyFloat dimer_1min1pls(PyInt n_orb1, PyInt n_orb2, Double* Rc1, Double* Rcca1, Double* Ra2, Double* Rcaa2, Double* h, Double* V1112, Double* V1222)
	{
	PyFloat H = 0;
	for (PyInt p1=0;  p1<n_orb1;  p1++)
		{
		for (PyInt q1=0;  q1<n_orb1;  q1++)
			{
			for (PyInt r1=0;  r1<n_orb1;  r1++)
				{
				for (PyInt s2=0;  s2<n_orb2;  s2++)
					{
					H += V1112[((p1*n_orb1 + q1)*n_orb1 + r1)*n_orb2 + s2] * Rcca1[(q1*n_orb1 + p1)*n_orb1 + r1] * Ra2[s2];
					}
				}
			}
		}
	for (PyInt p1=0;  p1<n_orb1;  p1++)
		{
		for (PyInt q2=0;  q2<n_orb2;  q2++)
			{
			for (PyInt r2=0;  r2<n_orb2;  r2++)
				{
				for (PyInt s2=0;  s2<n_orb2;  s2++)
					{
					H += V1222[((p1*n_orb2 + q2)*n_orb2 + r2)*n_orb2 + s2] * Rc1[p1] * Rcaa2[(q2*n_orb2 + s2)*n_orb2 + r2];
					}
				}
			}
		}
	H *= 2;
	for (PyInt p1=0;  p1<n_orb1;  p1++)
		{
		for (PyInt q2=0;  q2<n_orb2;  q2++)
			{
			H += h[p1*n_orb2 + q2] * Rc1[p1] * Ra2[q2];
			}
		}
	return H;
	}



PyFloat dimer_00(PyInt n_orb1, PyInt n_orb2, Double* Rca1, Double* Rca2, Double* V)
	{
	PyFloat H = 0;
	for (PyInt p1=0;  p1<n_orb1;  p1++)
		{
		for (PyInt q2=0;  q2<n_orb2;  q2++)
			{
			for (PyInt r1=0;  r1<n_orb1;  r1++)
				{
				for (PyInt s2=0;  s2<n_orb2;  s2++)
					{
					H += V[((p1*n_orb2 + q2)*n_orb1 + r1)*n_orb2 + s2] * Rca1[p1*n_orb1 + r1] * Rca2[q2*n_orb2 + s2];
					}
				}
			}
		}
	return 4*H;
	}



PyFloat dimer_1pls1min(PyInt n_orb1, PyInt n_orb2, Double* Ra1, Double* Rcaa1, Double* Rc2, Double* Rcca2, Double* h, Double* V2221, Double* V2111)
	{
	PyFloat H = 0;
	for (PyInt p2=0;  p2<n_orb2;  p2++)
		{
		for (PyInt q2=0;  q2<n_orb2;  q2++)
			{
			for (PyInt r2=0;  r2<n_orb2;  r2++)
				{
				for (PyInt s1=0;  s1<n_orb1;  s1++)
					{
					H += V2221[((p2*n_orb2 + q2)*n_orb2 + r2)*n_orb1 + s1] * Rcca2[(q2*n_orb2 + p2)*n_orb2 + r2] * Ra1[s1];
					}
				}
			}
		}
	for (PyInt p2=0;  p2<n_orb2;  p2++)
		{
		for (PyInt q1=0;  q1<n_orb1;  q1++)
			{
			for (PyInt r1=0;  r1<n_orb1;  r1++)
				{
				for (PyInt s1=0;  s1<n_orb1;  s1++)
					{
					H += V2111[((p2*n_orb1 + q1)*n_orb1 + r1)*n_orb1 + s1] * Rc2[p2] * Rcaa1[(q1*n_orb1 + s1)*n_orb1 + r1];
					}
				}
			}
		}
	H *= 2;
	for (PyInt p2=0;  p2<n_orb2;  p2++)
		{
		for (PyInt q1=0;  q1<n_orb1;  q1++)
			{
			H += h[p2*n_orb1 + q1] * Rc2[p2] * Ra1[q1];
			}
		}
	return H;
	}



PyFloat dimer_2pls2min(PyInt n_orb1, PyInt n_orb2, Double* Raa1, Double* Rcc2, Double* V)
	{
	PyFloat H = 0;
	for (PyInt p2=0;  p2<n_orb2;  p2++)
		{
		for (PyInt q2=0;  q2<n_orb2;  q2++)
			{
			for (PyInt r1=0;  r1<n_orb1;  r1++)
				{
				for (PyInt s1=0;  s1<n_orb1;  s1++)
					{
					H += V[((p2*n_orb2 + q2)*n_orb1 + r1)*n_orb1 + s1] * Rcc2[p2*n_orb2 + q2] * Raa1[s1*n_orb1 + r1];
					}
				}
			}
		}
	return H;
	}
