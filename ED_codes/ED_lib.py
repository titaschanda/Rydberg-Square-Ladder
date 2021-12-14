import os
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import sys
import copy
from itertools import combinations
from functools import reduce
import numba as nb
from numba import jit
from numba.typed import List

def myreduce(iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = sp.kron(value, element, format='csc')
    return value

### BINARY OPERATION TO BUILD THE OCCUPATION BASIS

@nb.njit() ### RETURNS SINGLE CHAIN OCCUPATION STATES FROM A LADDER OCCUPATION STATE
def split(a, L):
    b = 0
    c = 0
    for i in range(L):
        b += 2**(i)
        c += 2**(L + i)
    b = a & b
    c = (a & c)//(2**L)
    return b, c

@nb.njit() ### MERGES TWO SINGLE CHAIN OCCUPATION STATES INTO A LADDER OCCUPATION STATE
def join(b, c, L):
    return b + (c*(2**L))

@nb.njit() ### TRANSLATION TO THE RIGHT ON THE SINGLE CHAIN
def T1R(a, L):
    return (a >> 1) + (2**(L-1))*(a & 1)

@nb.njit() ### TRANSLATION TO THE LEFT ON THE SINGLE CHAIN
def T1L(a, L):
    return (a << 1) + ((a & (2**(L-1)))//(2**(L-1)))

@nb.njit() ### TRANSLATION TO THE RIGHT ON THE LADDER
def TR(a, L):
    b, c = split(a, L)
    b1 = T1R(b, L)
    c1 = T1R(c, L)
    return join(b1, c1, L)

@nb.njit() ### CHAIN REFLECTION
def CR(a, L):
    b, c = split(a, L)
    return join(c, b, L)

@nb.njit() ### GET THE NUMBER OF PARTICLES IN AN OCCUPATION STATE
def bos_num(a, L):
    nb = 0
    for i in range(L):
        nb += (a & 2**i) // (2**i)
    return nb

@nb.njit() ### GIVES THE OCCUPATION BASIS FOR A SINGLE CHAIN GIVEN THE FILLING AND THE CHAIN LENGTH
def leg_basis(nb, L):
    basis = [0]
    for i in range(2**L):
        if bos_num(i, L)==nb: basis +=[i]

    return basis[1:]

@nb.njit() ### GIVES THE OCCUPATION BASIS FOR THE LADDER GIVEN AN OCCUPATION ARRAY AND THE CHAIN LENGTH
def lad_basis(nb, L):
    basis = [0]

    b1 = leg_basis(nb[0], L)
    if nb[0]==nb[1]: b2 = b1
    else: b2 = leg_basis(nb[1], L)
    
    b2 = np.array(b2)
    for i in range(len(b1)):
        basis += list(b2 + (2**L)*b1[i])

    return basis[1:]


@nb.njit() ### GIVES FEEDBACK ON THE NUMBER OF PARTICLES IN AN OCCUPATION STATE
def check_num(a, L, nb):
    nb1, nb2 = nb
    b, c = split(a, L)
    if bos_num(b, L)!=nb1: 
        return a, -1
    else:
        if bos_num(c, L)!=nb2: 
            return a, -1
        else:
            return a, 1

    


### BUILDS MANY-BODY OPERATORS IN TERMS OF LOCAL OPERATORS AND THEIR LOCATIONS

def operator(ops, ind, L, full_basis):

    ### REORDER INDICES, USEFUL FOR PBC

    ind = np.array(ind)
    sites = ind[:, 0]
    legs = ind[:, 1]
    x = (sites + legs*L)
    ops = np.array(ops)

    ### TENSOR PRODUCT
  
    if x[1]<x[0]:
        v = x[1]
        x[1] = x[0]
        x[0] = v

        v = ops[1]
        ops[1] = ops[0]
        ops[0] = v

    if x[1] == x[0]:
        ops = np.dot(ops[1], ops[0])
        tensor = myreduce((sp.eye(2**(2*L-x[0]-1)), ops, sp.eye(2**x[0])))
    else:
        #print("op0", ops[0])
        #print("op1", ops[1])
        #tensor2 = reduce(np.kron, (np.eye(2**(2*L-x[1]-1)), ops[1].toarray(), np.eye(2**(x[1]-x[0]-1)), ops[0].toarray(), np.eye(2**x[0])))
        tensor = myreduce((sp.eye(2**(2*L-x[1]-1)), ops[1], sp.eye(2**(x[1]-x[0]-1)), ops[0], sp.eye(2**x[0])))
    

    tensor = tensor[:, full_basis]
    tensor = tensor[full_basis, :]

    return sp.csc_matrix(tensor)


def full_state(psi, basis, L):
    full_psi = np.zeros(2**(2*L))*(0+0j)
    full_psi[basis] = psi
    return full_psi
