
In order to run the tests, the following QodeApplications directories must be in the PYTHONPATH
    hermitian-XRCC
    Be-states
    frag-states
    StateSpaceOptimizer

Simply running pytest from inside the tests directory runs all tests, but a test can also be specified by filename.

It is recommendable to run with pytest -W ignore (unless you are in the mood to fix a bunch of deprecation messages).

Below are some special notes for specific tests

preliminary_test.py
    It is recommended to generate your own testdata before applying changes.  In order to do so,
    read the instructions at the bottom of .../hermitian-XRCC/mains/workflow.py.

xr_test.py
    This will not run without first generating an amount of data that cannot be checked in for the
    purposes of a test (and generating on the fly just for the test would take soooo long).
    To do this, run the following from inside .../frag-states
      mkdir rho
      ./driver.py Be2.in.py n_threads=<n>
      python decomps.py 0
    Then softlink .../frag-states/rho to inside .../hermitian-XRCC
