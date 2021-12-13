import numpy as np
from numba import njit

# Create np ops

@njit
def make_boson(n0):
    A = np.zeros((n0+1, n0+1), dtype = np.float64)
    N = np.zeros((n0+1, n0+1), dtype = np.float64)
    Idb = np.zeros((n0+1, n0+1), dtype = np.float64)
    for ii in range(n0+1):
        N[ii, ii] = ii
        Idb[ii, ii] = 1.0
        if (ii < n0):
            A[ii, ii+1] = np.sqrt(ii+1)
    Adag = A.T
    return Idb, A, Adag, N

@njit 
def make_QBoson(n0):
    
    Id0, A0, Adag0, N0 = make_boson(n0)
    
    A = np.kron(A0, Id0)
    Adag = np.kron(Adag0, Id0)
    NA = np.kron(N0, Id0)
    
    B = np.kron(Id0, A0)
    Bdag = np.kron(Id0, Adag0)
    NB = np.kron(Id0, N0)
    
    return A, Adag, NA, B, Bdag, NB


@njit
def BoseFilling(i, j, n0):
    d = n0+1    
    return i*d + j
