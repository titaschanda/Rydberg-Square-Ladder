from ED_lib import *

### FUNCTIONS TO BUILD HAMILTONIAN TERMS

def rung(L, basis): ### U n_i+ n_i-
    n = sp.csc_matrix([[0, 0], [0, 1]])
    H = 0
    for i in range(L):
        H += operator([n, n], [[i, 0], [i, 1]], L, basis)
    print("Rung term ready!")
    return H


def hop(L, basis): ### t(b_i b^dag_i+1 + h.c.)
    bp = sp.csc_matrix([[0, 0], [1, 0]])
    bm = sp.csc_matrix([[0, 1], [0, 0]])
    H = 0
    for i in range(L):
        for s in range(2):
            H += operator([bp, bm], [[i, s], [(i+1)%L, s]], L, basis)
            H += operator([bm, bp], [[i, s], [(i+1)%L, s]], L, basis)
    print("Hopping term ready!")
    return H


def dress(L, basis, rc): ### V n_i n_j , j<=i+rC
    n = sp.csc_matrix([[0, 0], [0, 1]])
    H = 0
    for i in range(L):
        for s in range(2):
            for j in range(1, rc + 1):
                H += operator([n, n], [[i, s], [(i+j)%L, s]], L, basis)
    return H
    print("Dressing term ready!")