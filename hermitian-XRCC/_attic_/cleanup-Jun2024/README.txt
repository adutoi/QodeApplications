There are now essentially 5 "main" files depending on what one is doing.
All of them run on top of the same underlying infrastructure code.

With respect to the prior content of this file, which was a "Usage:" statement,
that should be the responsibility of comments in the respective "mains".

===================
test.py was last touched by Marco, and to keep off each other's toes, this file is his now.

===================
The four files presently in mains/ should be thought of as Tony's.  They do either an XR or an XR[1] (*)
calculation, using either CI or CC methodology (and getting the same answer to within convergence threshold,
except for a region where XR[1] appears to be fundamentaly unstable).  These are still evolving.

(*) XR[1] is the new nomenclature (for now) for what we have been calling XR'.  The reasons we need a
new name are twofold: (a) to avoid ambiguity with what was called XR' in the first MolPhys paper, using
the brute-force computation of the dimer Hamiltonian; (b) because what we are doing now only approaches
XR' as we go to infinite order in sigma, so we need to denote the order of sigma somehow, which was
first-order in this case (and, so far, we are computing S^{-1} by taking S to first order in sigma
and then fully inverting it numerically ... we are not expanding around S=diag(1) and truncating).
This gives us XR[0]=XR and XR[inf]=XR', which is pretty tidy.
