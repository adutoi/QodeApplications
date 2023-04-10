(Psi4) (bash) [adutoi@medusa custom]$ python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> np.save("Z_0e.npy", np.identity(1))
>>> np.save("Z_1e.npy", np.identity(4))
>>> np.save("Z_2e.npy", np.identity(6))
>>> np.save("Z_3e.npy", np.identity(4))
>>> np.save("Z_4e.npy", np.identity(1))
