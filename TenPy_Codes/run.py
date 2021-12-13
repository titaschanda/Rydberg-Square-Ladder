from model import *
import sys

import pickle
import gzip


## ********** Input Parameters ********************


MaxChi = int(sys.argv[1]) # Bond dimension \chi
U = np.float64(sys.argv[2]) # system parameter U
V = np.float64(sys.argv[3]) # system parameter V

## ************************************************


## ********** DMRG parameters *********************

MAXSW = 2000 ## Maximum number of DMRG sweeps
MinSw = 1000 ## Minimum number of DMRG sweeps

ND = 20
disa_aft = MinSw//2 + MinSw//5
Mix_high = 1
Mix_low = 1e-5

dSw = np.linspace(0, MinSw//2, ND)
dCh = np.linspace(5, MaxChi, ND)
Ch_list = {}
for ii in range(ND):
    Ch_list[int(dSw[ii])] = int(dCh[ii])

decay = (Mix_high/Mix_low)**(1/disa_aft)


dmrg_params2 = {    
    'mixer': True, 
    'max_E_err': 1.e-12,
    'max_S_err': 1.e-10,
    'max_sweeps': MAXSW,
    'min_sweeps' : MinSw,
    'trunc_params': {
        'svd_min': 1.e-10
    },
    'chi_list' : Ch_list,    
    'mixer_params' : {
        'amplitude' : Mix_high,
        'decay' : decay,
        'disable_after' : disa_aft
    },
    'verbose' : True
}

## ***********************************************************


## ***************** Actual Simulation part ******************

LAct = 10 # length of the unit cell
n0 = 1 # hardcore limit


tenpy.tools.process.mkl_set_nthreads(1)

    
model_params = dict(L=LAct, n0=n0, U=U, V=V, t=1.0,
                    bc_MPS='infinite')

M = SpinfullBose(model_params)

State = np.array([BoseFilling(1, 1, n0), BoseFilling(0, 0, n0), BoseFilling(1, 1, n0),
                  BoseFilling(0, 0, n0), BoseFilling(0, 0, n0)] * (LAct//5))

psi = MPS.from_product_state(M.lat.mps_sites(), State, bc=M.lat.bc_MPS)

eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params2)
E, psi = eng.run()
psi.canonical_form()


## ********* Save the MPS to the disk ***************************

Name = "{:.2f}".format(U) + "_" + "{:.2f}".format(V) + "_psi_" + str(MaxChi) + ".pkl"
with gzip.open(Name, 'wb') as f:
    pickle.dump(psi, f)

## **************************************************************
