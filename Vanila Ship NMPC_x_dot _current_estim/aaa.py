import math
import numpy as np

a = np.array([1.25174589e-02, 1.18475367e-02, 9.53060081e-03, 1.24046583e-02,1.19666792e-02, 9.53090241e-03, 4.99995362e+01, 4.99997228e+01, 1.24039744e-02])
print(a[8])
print(a[0]*math.cos(a[5]) - a[1]*math.sin(a[5]))


