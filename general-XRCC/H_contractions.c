/*   (C) Copyright 2018, 2019 Anthony D. Dutoi and Yuhong Liu
 *
 *   This file is part of QodeApplications.
 *
 *   QodeApplications is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   QodeApplications is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with QodeApplications.  If not, see <http://www.gnu.org/licenses/>.
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

PyFloat monomer_1e(PyInt n_orb, Double* Rca, Double* h)
	{
	PyFloat H = 0;
	for (PyInt p=0;  p<n_orb;  p++)
		{
		for (PyInt q=0;  q<n_orb;  q++)
			{
			H += h[p*n_orb + q] * Rca[p*n_orb + q];
			}
		}
	return H;
	}

PyFloat monomer_2e(PyInt n_orb, Double* Rccaa, Double* V)
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
	return H;
	}

PyFloat monomer_extPot(PyInt n_orb, Double* Rca, Double* h)
	{
	PyFloat H = 0;
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



PyFloat dimer_1min1pls_1e(PyInt n_orb1, PyInt n_orb2, Double* Rc1, Double* Ra2, Double* h)
	{
	PyFloat H = 0;
	for (PyInt p1=0;  p1<n_orb1;  p1++)
		{
		for (PyInt q2=0;  q2<n_orb2;  q2++)
			{
			H += h[p1*n_orb2 + q2] * Rc1[p1] * Ra2[q2];
			}
		}
	return H;
	}

PyFloat dimer_1min1pls_2e(PyInt n_orb1, PyInt n_orb2, Double* Rc1, Double* Rcca1, Double* Ra2, Double* Rcaa2, Double* V1112, Double* V1222)
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
	return 2*H;
	}



PyFloat dimer_ExEx(PyInt n_orb1, PyInt n_orb2, Double* Rca1, Double* Rca2, Double* V)
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



PyFloat trimer_2min1pls1pls(PyInt n_orb1, PyInt n_orb2, PyInt n_orb3, Double* Rcc1, Double* Ra2, Double* Ra3, Double* V)
	{
        PyFloat Rqp, Rqp_Rr;
	PyFloat H = 0;
	for (PyInt p1=0;  p1<n_orb1;  p1++)
		{
		for (PyInt q1=0;  q1<n_orb1;  q1++)
			{
			Rqp = Rcc1[q1*n_orb1 + p1];
			for (PyInt r2=0;  r2<n_orb2;  r2++)
				{
				Rqp_Rr = Rqp * Ra2[r2];
				for (PyInt s3=0;  s3<n_orb3;  s3++)
					{
					H += V[((p1*n_orb1 + q1)*n_orb2 + r2)*n_orb3 + s3] * Rqp_Rr * Ra3[s3];
					}
				}
			}
		}
	return 2*H;
	}



PyFloat trimer_2pls1min1min(PyInt n_orb1, PyInt n_orb2, PyInt n_orb3, Double* Raa1, Double* Rc2, Double* Rc3, Double* V)
	{
        PyFloat Rr, Rr_Rs;
	PyFloat H = 0;
	for (PyInt r2=0;  r2<n_orb2;  r2++)
		{
		Rr = Rc2[r2];
		for (PyInt s3=0;  s3<n_orb3;  s3++)
			{
			Rr_Rs = Rr * Rc3[s3];
			for (PyInt p1=0;  p1<n_orb1;  p1++)
				{
				for (PyInt q1=0;  q1<n_orb1;  q1++)
					{
					H += V[((r2*n_orb3 + s3)*n_orb1 + p1)*n_orb1 + q1] * Rr_Rs * Raa1[q1*n_orb1 + p1];
					}
				}
			}
		}
	return 2*H;
	}



PyFloat trimer_Ex1min1pls(PyInt n_orb1, PyInt n_orb2, PyInt n_orb3, Double* Rca1, Double* Rc2, Double* Ra3, Double* V)
	{
        PyFloat Rr, Rr_Rpq;
	PyFloat H = 0;
	for (PyInt p1=0;  p1<n_orb1;  p1++)
		{
		for (PyInt r2=0;  r2<n_orb2;  r2++)
			{
			Rr = Rc2[r2];
			for (PyInt q1=0;  q1<n_orb1;  q1++)
				{
				Rr_Rpq = Rr * Rca1[p1*n_orb1 + q1];
				for (PyInt s3=0;  s3<n_orb3;  s3++)
					{
					H += V[((p1*n_orb2 + r2)*n_orb1 + q1)*n_orb3 + s3] * Rr_Rpq * Ra3[s3];
					}
				}
			}
		}
	return 4*H;
	}
