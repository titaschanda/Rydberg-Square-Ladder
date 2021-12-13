import tenpy
from tenpy.algorithms import dmrg
import tenpy.linalg.np_conserved as npc

from tenpy.models.model import MPOModel, CouplingMPOModel
from tenpy.models.lattice import Site
from tenpy.tools.params import asConfig
from tenpy.networks.mps import MPS

from helper_func import *


class SpinfullBose(CouplingMPOModel):
    
    default_lattice = "Chain"
    force_default_lattice = True
    
    def init_sites(self, model_params):
        
        n0 = model_params.get("n0", 1)
        
        A, Adag, NA, B, Bdag, NB = make_QBoson(n0)

        conserve = model_params.get("conserve", True)
        qlist = list(zip(np.diag(NA).astype(int), np.diag(NB).astype(int)))
        
        if conserve:            
            leg = npc.LegCharge.from_qflat(npc.ChargeInfo([1, 1], ["Na", "Nb"]), qlist)
        else:
            leg = npc.LegCharge.from_trivial((n0+1)**2)

        site = Site(leg, qlist, 
                     A=A, Adag=Adag, NA=NA, B=B, Bdag=Bdag, NB=NB)
        return site
    
    
    def init_terms(self, model_params):
        
        L = model_params.get("L", 20)
        
        
        t = model_params.get("t", 1.0)
        U = model_params.get("U", 1.0)
        V = model_params.get("V", 1.0)        
        
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(U, u, 'NA NB')

            
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Adag', u2, 'A', dx, plus_hc=True)
            self.add_coupling(-t, u1, 'Bdag', u2, 'B', dx, plus_hc=True)
            
            self.add_coupling(V, u1, 'NA', u2, 'NA', dx, plus_hc=False)
            self.add_coupling(V, u1, 'NA', u2, 'NA', 2*dx, plus_hc=False)
            
            self.add_coupling(V, u1, 'NB', u2, 'NB', dx, plus_hc=False)
            self.add_coupling(V, u1, 'NB', u2, 'NB', 2*dx, plus_hc=False)
