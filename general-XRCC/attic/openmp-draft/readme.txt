
this stuff originates from an attempt to use openMP on the C side to replace multiprocessing on the python side.

the main advantage of this would be that test_H.py would look almost exactly like the one that Vivek's group is using,
so that future mergers should be seamless.

however the performance sucked and I think it is because the openMP version gets memory bound in a way that the multiprocessing
version does not because the multiprocessing version recomputes all the rhos (which I *thought* was stupid).
 . . . so I am back to using the multiprocessing version.

However, as of the writing of this file, the versions of test_H.py and H_contractions.c (which are the only files that needed to change)
that are in use are identical to the ".old" versions in this directory.  These can be directly swapped for the ".new" versions
and the code still runs (just slower).

So, this means that, when it is time to integrate with the (hopefully faster) code of vivek's group, I should just be able
to diff the versions of test_H.py and H_contractions.c that are then in use with the ".old" versions here.  If there are
no changes, then they presumably can still be swapped out with the ".new" versions, which are closer to the code from vivek's group,
and merging should be a bit simpler.

