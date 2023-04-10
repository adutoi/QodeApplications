  655  cd atomic_states_new/H/4-6-4/thresh\=1e-100/
  661  mkdir custom
  662  cp 4.5/Z_*.npy custom/
  664  cd custom/
  668  python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> np.save("Z_0e.npy", np.array([[1.]]))
>>> np.save("Z_4e.npy", np.array([[1.]]))
