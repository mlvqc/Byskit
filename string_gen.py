import numpy as np
n = 3
a = np.arange(0,2**n)
for i in a:
    print(i)
    print(np.binary_repr(i,width = n))
    print('-----')
