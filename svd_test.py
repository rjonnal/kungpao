import numpy as np

a = np.matrix([[2, -1, 0],[4,3,-2]])
a = np.random.randn(10,6)
ai1 = np.linalg.pinv(a)


u, s, vt = np.linalg.svd(a, full_matrices=True)
term1 = vt.T
term2 = np.zeros(a.T.shape)
term2[:len(s),:len(s)] = np.linalg.pinv(np.diag(s))
term3 = u.T
ai2 = np.dot(np.dot(term1,term2),term3)

print ai1
print
print ai2
print

print np.allclose(ai1,ai2)
