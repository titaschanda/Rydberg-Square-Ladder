from ED_lib import *
from hamiltonian_terms import *
from fidelity import *

### BRIEF TUTORIAL FOR EXACT DIAGONALIZATION CODES


### PARAMETERS OF THE MODEL FOR OCCUPATION BASIS GENERATION

L = 10
n_b = np.array([4, 4])
rc = 2

basis = lad_basis(n_b, L)
print("Basis ready!")

### GENERATION OF THE HAMILTONIAN TERMS

Ht = hop(L, basis)
HU = rung(L, basis)
HV = dress(L, basis, rc)

Hlist = [Ht, HU, HV]

### LOW-LYING ENERGY SPECTRUM GIVEN THE COUPLINGS

U = 0
V = 0

trunc = 2

H = Ht + U * HU + V * HV
E, psi = spa.eigsh(H, which = "SA", k = trunc)
print("Diagonalized!")

gap = E[1] - E[0]
print("Energy gap = ", gap)
print("Ground state energy density = ", E[0]/(2*L))

### FIDELITY CALCULATION

Uval = [0, 2]
Vval = [0, 5]

fid_val = fid(Hlist, Uval, Vval)
print("Fidelity = ", fid_val)

