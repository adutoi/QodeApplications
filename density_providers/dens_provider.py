#    (C) Copyright 2026 Marco Bauer
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



class DensBackend:
    def __init__(self, backend):
        self.backend = backend

    def init_backend(self, frags):
        if self.backend == "hummr":
            from hummr_dens.pybind_interface import HummrXREngine

            self.engine = HummrXREngine()
            #engine.configure()

            # frags should be [frag_name, input_file, roots, xr_level]
            ret = []
            for frag in frags:
                frag_engine = self.engine.add_fragment(
                    frag[0],
                    xr_level=frag[3],
                )

                """
                input_file = be.prepare_input(
                    scratch_dir="scratch",
                    filename="Be.inp",

                    calc_type="CASSCF",
                    charge=0,
                    multiplicities=[1, 3],

                    orb_guess="/home/marco/hummr_tests/Be_mos.C0",

                    ints="RI",
                    basis="6-31G",
                    aux_basis="def2-JK",

                    n_electrons=4,
                    n_orbitals=9,
                    nroots=[4, 2],

                    ci_solver="FCI",
                    max_iter=0,
                    orb_step="SuperCIPTDIIS",
                    switch_orb_step="DIIS",

                    geometry=["Be 0 0 0"],
                )
                """

                mo_coeffs = frag_engine.initialize(frag[1])

                states = frag_engine.get_states(frag[2])

                ret.append([mo_coeffs, states])
            return ret

    def build_densities(
            self,
            #frag_ind,
            #dens_builder_stuff,
            frag_name,
            xr_level,
            bra_states,
            ket_states,
            #n_threads,
            ):
        """
        Build all fragment density tensors required by get_xr_H.

        state_coeffs[chg] has shape
            (n_states, n_csf)

        Returns the density object expected by get_xr_H.
        """
        if self.backend == "in_house":
            pass
            #from densities import build_tensors  # from Be-states (set PYTHONPATH for this to work)
            #return build_tensors(*dens_builder_stuff[frag_ind][:-1],
            #                      options=dens_builder_stuff[frag_ind][-1],
            #                      n_threads=n_threads)
        elif self.backend == "hummr":
            # only pybind11 interface available here, but if you need it, there is an additional
            # JSON import in the hummr directory, as well as in the Be-states.dens_from_hummr.py
            return self.engine.fragments[frag_name].compute_densities(
                            bra=bra_states,
                            ket=ket_states,
                            xr_level=xr_level,
                        )
        else:
            raise NotImplementedError

    def finalize_all(self):
        if self.backend == "hummr":
            for hummr_instance in self.engine.fragments.values():
                hummr_instance.finalize()
        else:
            pass
