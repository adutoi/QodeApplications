  655  cd atomic_states_new/data
  657  python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> np.save("configs_0e-stable.npy", np.array([[0,3]]))
>>> np.save("configs_4e-stable.npy", np.array([[0,1,2,3,4,5]]))
