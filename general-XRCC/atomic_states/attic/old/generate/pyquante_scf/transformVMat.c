/*   (C) Copyright 2018 Anthony D. Dutoi
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
#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>

long long idx_M_first_x(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx1*size + idx2; }

long long idx_M_first_y(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx3*size + idx; }


long long idx_M_second_x(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx1*size + idx2; }

long long idx_M_second_y(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx*size + idx4; }


long long idx_M_third_x(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx1*size + idx; }

long long idx_M_third_y(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx3*size + idx4; }



long long idx_M_fourth_x(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx*size + idx2; }

long long idx_M_fourth_y(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx3*size + idx4; }


long long idx_U_first_x(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx; }

long long idx_U_first_y(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx4; }


long long idx_U_second_x(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx; }

long long idx_U_second_y(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx3; }


long long idx_U_third_x(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx; }

long long idx_U_third_y(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx2; }


long long idx_U_fourth_x(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx; }

long long idx_U_fourth_y(long long size, long long idx1, long long idx2, long long idx3, long long idx4, long long idx)
{ return idx1; }



void array_fill(double* array, double value, long long dim)
{
    long long i, j;
    for(i=0; i<dim; i++)
    {
        for(j=0; j<dim; j++)
        {
            array[i*dim+j] = value;
        }
    }
}

void transLoop1( double* Mat, double* U, double* Ret_Mat, long long mat_dim )
{
    long long stride = mat_dim * mat_dim;
    long long idx, idx1, idx2, idx3, idx4;
    for (idx1=0; idx1<mat_dim; idx1++)
    {
        for (idx2=0; idx2<mat_dim; idx2++)
        {
            for (idx3=0; idx3<mat_dim; idx3++)
            {
                for (idx4=0; idx4<mat_dim; idx4++)
                {
                    for(idx=0; idx<mat_dim; idx++)
                    {
                        Ret_Mat[(idx1*mat_dim+idx2)*stride + idx3*mat_dim+idx4] += \
                            Mat[idx_M_first_x(mat_dim,idx1,idx2,idx3,idx4,idx) * stride + idx_M_first_y(mat_dim,idx1,idx2,idx3,idx4,idx)] *\
                            U[  idx_U_first_x(mat_dim,idx1,idx2,idx3,idx4,idx) * mat_dim + idx_U_first_y(mat_dim,idx1,idx2,idx3,idx4,idx)];
                    }
                }
            }
        }
    }
}

void transLoop2( double* Mat, double* U, double* Ret_Mat, long long mat_dim )
{
    long long stride = mat_dim * mat_dim;
    long long idx, idx1, idx2, idx3, idx4;
    for (idx1=0; idx1<mat_dim; idx1++)
    {
        for (idx2=0; idx2<mat_dim; idx2++)
        {
            for (idx3=0; idx3<mat_dim; idx3++)
            {
                for (idx4=0; idx4<mat_dim; idx4++)
                {
                    for(idx=0; idx<mat_dim; idx++)
                    {
                        Ret_Mat[(idx1*mat_dim+idx2)*stride + idx3*mat_dim+idx4] += \
                            Mat[idx_M_second_x(mat_dim,idx1,idx2,idx3,idx4,idx) * stride + idx_M_second_y(mat_dim,idx1,idx2,idx3,idx4,idx)] *\
                            U[  idx_U_second_x(mat_dim,idx1,idx2,idx3,idx4,idx) * mat_dim + idx_U_second_y(mat_dim,idx1,idx2,idx3,idx4,idx)];
                    }
                }
            }
        }
    }

}

void transLoop3( double* Mat, double* U, double* Ret_Mat, long long mat_dim )
{
    long long stride = mat_dim * mat_dim;
    long long idx, idx1, idx2, idx3, idx4;
    for (idx1=0; idx1<mat_dim; idx1++)
    {
        for (idx2=0; idx2<mat_dim; idx2++)
        {
            for (idx3=0; idx3<mat_dim; idx3++)
            {
                for (idx4=0; idx4<mat_dim; idx4++)
                {
                    for(idx=0; idx<mat_dim; idx++)
                    {
                        Ret_Mat[(idx1*mat_dim+idx2)*stride + idx3*mat_dim+idx4] += \
                            Mat[idx_M_third_x(mat_dim,idx1,idx2,idx3,idx4,idx) * stride + idx_M_third_y(mat_dim,idx1,idx2,idx3,idx4,idx)] *\
                            U[  idx_U_third_x(mat_dim,idx1,idx2,idx3,idx4,idx) * mat_dim + idx_U_third_y(mat_dim,idx1,idx2,idx3,idx4,idx)];
                    }
                }
            }
        }
    }

}

void transLoop4( double* Mat, double* U, double* Ret_Mat, long long mat_dim )
{
    long long stride = mat_dim * mat_dim;
    long long idx, idx1, idx2, idx3, idx4;
    for (idx1=0; idx1<mat_dim; idx1++)
    {
        for (idx2=0; idx2<mat_dim; idx2++)
        {
            for (idx3=0; idx3<mat_dim; idx3++)
            {
                for (idx4=0; idx4<mat_dim; idx4++)
                {
                    for(idx=0; idx<mat_dim; idx++)
                    {
                        Ret_Mat[(idx1*mat_dim+idx2)*stride + idx3*mat_dim+idx4] += \
                            Mat[idx_M_fourth_x(mat_dim,idx1,idx2,idx3,idx4,idx) * stride + idx_M_fourth_y(mat_dim,idx1,idx2,idx3,idx4,idx)] *\
                            U[  idx_U_fourth_x(mat_dim,idx1,idx2,idx3,idx4,idx) * mat_dim + idx_U_fourth_y(mat_dim,idx1,idx2,idx3,idx4,idx)];
                    }
                }
            }
        }
    }
}


double* transform_V(double* Mat, double* U, long long mat_dim)
{
    double* Ret_Mat  = (double*) malloc( sizeof(double) * mat_dim * mat_dim * mat_dim * mat_dim );
    double* temp_Mat = (double*) malloc( sizeof(double) * mat_dim * mat_dim * mat_dim * mat_dim );
    // Fill up arrays with 0
    array_fill( temp_Mat , 0.0, mat_dim*mat_dim );
    array_fill( Ret_Mat  , 0.0, mat_dim*mat_dim );
    // Transformation Loop 1 and 2
    transLoop1( Mat      , U, temp_Mat, mat_dim );
    transLoop2( temp_Mat , U, Ret_Mat , mat_dim );
    array_fill( temp_Mat , 0.0, mat_dim*mat_dim );
    // Transformation Loop 3
    transLoop3( Ret_Mat  , U, temp_Mat , mat_dim );
    array_fill( Ret_Mat  , 0.0, mat_dim*mat_dim );
    // Transformation Loop 4
    transLoop4( temp_Mat , U, Ret_Mat  , mat_dim );

    free(temp_Mat);
    return Ret_Mat;
}


PyObject* transformV(PyObject *self, PyObject *args)  // This is the wrapper to C functions
{
    PyObject* Py_Mat;
    PyObject* Py_U;
    // PyObject* Ret;
    

    if (!PyArg_ParseTuple(args, "OO", &Py_Mat, &Py_U))
    {
        puts("PyArg_ParseTuple Function Failed.");
        return NULL;
    }

    if (Py_Mat == NULL || Py_U == NULL)
    {
        puts("Failed to Parse the Input Matrices.");
        return NULL;
    }

    PyObject* Np_Mat  =  PyArray_FROM_OTF(Py_Mat, NPY_DOUBLE, NPY_ARRAY_DEFAULT); 
    PyObject* Np_U    =  PyArray_FROM_OTF(Py_U  , NPY_DOUBLE, NPY_ARRAY_DEFAULT); 
    long long mat_dim = (long long) PyArray_DIM(Py_U, 0);  // 0-th dimension size.

    if (Np_Mat == NULL || Np_U == NULL)
    {
        // decrease reference counts for these...
        PyArray_XDECREF( (PyArrayObject*)Np_Mat );
        PyArray_XDECREF( (PyArrayObject*)Np_U );
        puts("Failed to Convert to Numpy Array Types.");
        return NULL;
    }

    /*
    if ((PyArray_NDIM(Np_Mat) != 2) && (PyArray_NDIM(Np_U) != 2 ))
    {
        puts("Must Be Two-D Matrix to Transform");
        return NULL;
    } */

    // Now Get the raw data in C arrays.
    double* Mat = (double*)PyArray_DATA( Np_Mat );
    double* U   = (double*)PyArray_DATA( Np_U );

    if (Mat == NULL || U == NULL)
    {
        PyArray_XDECREF( (PyArrayObject*)Np_Mat );
        PyArray_XDECREF( (PyArrayObject*)Np_U );
        puts("Retrieving Mat or U Data Failed.");
        return NULL;
    }

    // printf("U dim = %lld\n", mat_dim);
    // printf("Mat[0,0] = %lf, U[0,0] = %lf\n", Mat[0], U[0] );
    // puts("ALL GOOD BEFORE TRANSFORMATION!");
    double* Ret_Mat = transform_V(Mat, U, mat_dim);

    // long long x,y;
    // for (x=0;x<mat_dim*mat_dim;x++)
    // {
    //     for (y=0;y<mat_dim*mat_dim;y++)
    //     {
    //         printf("Ret_M[%lld,%lld] = %lf\n", x,y, Ret_Mat[x*mat_dim*mat_dim +y]);
    //     }
    // }


    npy_intp Ret_dims[2]  = {mat_dim*mat_dim, mat_dim*mat_dim}; 
    PyObject* Np_Ret_Mat  = PyArray_SimpleNewFromData(2, Ret_dims, NPY_DOUBLE, Ret_Mat);

    // Cleaning Up (decrease reference count)
    PyArray_XDECREF( (PyArrayObject*) Np_Mat);
    PyArray_XDECREF( (PyArrayObject*) Np_U);   

    // Build a Numpy Matrix with Ret_Mat
    return Np_Ret_Mat;
}


// Module's Method Table
static PyMethodDef TransformVMethods[] = {
    {"transformV",  transformV, METH_VARARGS, "Transform the electronic repulsion matrix."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef transformVMatmodule = {
   PyModuleDef_HEAD_INIT,
   "transformVMat",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
   TransformVMethods
};


PyMODINIT_FUNC PyInit_transformVMat(void)
{
    import_array();
    return PyModule_Create(&transformVMatmodule);
}




