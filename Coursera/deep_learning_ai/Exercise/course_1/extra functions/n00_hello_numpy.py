import numpy as np

a = np.random.randn(5)
print(a)
print(a.shape)
print(a.T)
print(np.dot(a, a.T))

a = np.random.random