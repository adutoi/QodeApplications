
  474  python -u test_HbigCC.py 2 4.5 thresh=1e-6/all_eigen all 1 0 0 1 all >& std3.out &
  486  mkdir 1-3
  487  mv u.npy S.npy vh.npy std3.out 1-3/

  496  python -u test_HbigCC.py 2 4.5 thresh=1e-6/all_eigen 0 1 all all 1 0 >& std3.out &
  514  mkdir 3-1
  515  mv u.npy S.npy vh.npy std3.out 3-1/

  523  mkdir anion_states
  524  mv 1-3 3-1 anion_states/
  533  mv anion_states atomic_states/H/16-115-550/thresh\=1e-6/

  534  cd atomic_states/H/16-115-550/thresh\=1e-6/
  536  mkdir anion_states/A
  537  mkdir anion_states/B
  538  cp 4.5/Z_2e.npy anion_states/A/
  539  cp 4.5/Z_2e.npy anion_states/B/
  541  cp all_eigen/Z_1e.npy anion_states/A/
  542  cp all_eigen/Z_1e.npy anion_states/B/


  540  cp all_eigen/Z_3e.npy anion_states/
  544  cd anion_states/
  546  python

>>> import numpy as np
>>> 
>>> Z = np.load("Z_3e.npy")
>>> A = np.load("3-1/u.npy")
>>> B = np.load("1-3/vh.npy")
>>> 
>>> Z.shape
(560, 550)
>>> A.shape
(550, 16)
>>> B.shape
(16, 550)
>>> 
>>> ZA = Z.dot(A)
>>> ZB = Z.dot(B.T)
>>> np.save("A/Z_3e.npy", ZA)
>>> np.save("B/Z_3e.npy", ZB)


  571  rm A/Z_2e.npy 
  572  rm B/Z_2e.npy 
  573  cp ../all_eigen/Z_2e.npy A
  574  cp ../all_eigen/Z_2e.npy B
