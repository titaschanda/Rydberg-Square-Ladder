from ED_lib import *
from hamiltonian_terms import *

### COMPUTE THE FIDELITY BETWEEN TWO POINTS OF THE PHASE DIAGRAM

def fid(H, U, V):

        ### FEED WITH A LIST CONTAINING THE HAMILTONIAN TERMS
        
        Ht, HU, HV = H
        
        H1 = Ht + U[0]*HU + V[0]*HV
        H2 = Ht + U[1]*HU + V[1]*HV

        E, psi1 = spa.eigsh(H1, which = "SA", k = 1); psi1 = psi1[:, 0]
        E, psi2 = spa.eigsh(H2, which = "SA", k = 1); psi2 = psi2[:, 0]

        return abs(np.vdot(psi1, psi2))
