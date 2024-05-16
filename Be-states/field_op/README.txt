There have been a few attempts to improve the performance and capabilities of field-op.c

field_op-restrict.c
Be able to put in index restrictions to avoid looping over a bunch of zeros for special kinds of operators.
I think this is done but never tested?

field_op-fast.c
Orthogonal to the above, I wondered if the state-finding kernel was really the big slow-down so I made
a look-up table version that works only when configurations fit into a single int, but it did not
seem to help (?!).

field_op-fast-wo-core.c
On top of the above, I tried to find a quick way to get densities computed faster by not considering
core orbitals.  Also did not help.

These files also needed to be modified to cooperate with field_op-fast-wo-core.c
field_op.py
field_op_ham.py
densities.py
main-May2024-monomerFCIrho.py
