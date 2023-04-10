  505  cd atomic_states/H/16-115-550/thresh=1e-6
  509  cp -r common-1e-6-wo_3.5 common-1e-6-wo_3.5-cat2
  510  cd common-1e-6-wo_3.5-cat2
  512  python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> np.save("Z_0e.npy", np.array([[1.]]))

