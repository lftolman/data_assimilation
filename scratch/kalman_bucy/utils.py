import numpy as np

def extract_terms(f,t,uI,uII):
    eps=1e-6
    d = len(uII)
    f_base = f(t, uI, uII)
    J = np.zeros((np.atleast_1d(f_base).shape[0], d))
    for i in range(d):
        uII_eps = np.array(uII, dtype=np.complex128)
        uII_eps[i] += eps
        diff = (f(t, uI, uII_eps) - f_base) / eps
        J[:, i] = diff.flatten()
    a1 = J
    a0 = f_base - (a1 @ uII).reshape(f_base.shape)
    return a0, a1