/*   (C) Copyright 2026 Marco Bauer
 *
 *   This file is part of QodeApplications.
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
 *   along with QodeApplications.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "density_engine.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(pylible_hummr, m)
{
    m.doc() =
        "Python interface to HUMMR density construction.";

    py::class_<hummr::python::DensityEngine>(
        m,
        "DensityEngine"
    )
        .def(
            py::init<>()
        )

        .def(
            "initialize",
            [](
                hummr::python::DensityEngine& self,
                //py::array_t<double,
                //            py::array::c_style |
                //            py::array::forcecast> two_el_ints,
                //py::array_t<double,
                //            py::array::c_style |
                //            py::array::forcecast> one_el_ints,
                const std::vector<int>& charges,
                const std::vector<int>& multiplicities,
                const std::map<int, int>& n_electrons,
                const std::map<
                    std::pair<int, int>,
                    int
                >& n_roots,
                int n_orbitals)
            {
                //DArray4D two =
                //    numpy_to_darray4d(two_el_ints);

                //DMatrix one =
                //    numpy_to_dmatrix(one_el_ints);

                self.initialize(
                    two,
                    one,
                    charges,
                    multiplicities,
                    n_electrons,
                    n_roots,
                    n_orbitals
                );
            },
            //py::arg("two_el_ints"),
            //py::arg("one_el_ints"),
            py::arg("charges"),
            py::arg("multiplicities"),
            py::arg("n_electrons"),
            py::arg("n_roots"),
            py::arg("n_orbitals")
        )

        .def(
            "get_states",
            [](
                const hummr::python::DensityEngine& self)
            {
                return state_map_to_python(
                    self.get_states()
                );
            }
        )

        .def(
            "compute_densities",
            [](
                hummr::python::DensityEngine& self,
                const py::dict& bra_states,
                const py::dict& ket_states,
                int xr_level)
            {
                auto bra =
                    python_to_state_map(bra_states);

                auto ket =
                    python_to_state_map(ket_states);

                auto densities =
                    self.compute_densities(
                        bra,
                        ket,
                        xr_level
                    );

                return densities_to_python(
                    densities
                );
            },
            py::arg("bra_states"),
            py::arg("ket_states"),
            py::arg("xr_level") = 0
        );
}

