import numpy as np

def extract_terms(f,t,uI,uII):
    eps=1e-5
    d = len(uII)
    J = np.zeros((np.atleast_1d(f(t, uI, uII)).shape[0], d))
    for i in range(d):
        uII_eps = np.array(uII, dtype=float)
        uII_eps[i] += eps
        diff = (f(t, uI, uII_eps) - f(t, uI, uII)) / eps
        J[:, i] = diff
    a1 = J
    a0 = f(t, uI, uII) - a1 @ uII
    return a0, a1