from model import *
import sys

import pickle
import gzip

## ********** Input Parameters ********************
chi = int(sys.argv[1]) # Bond dimension \chi
U = np.float64(sys.argv[2]) # system parameter U
V = np.float64(sys.argv[3]) # system parameter V
## ************************************************


## ********** Read MPS ***************************
with gzip.open("{:.2f}".format(U) + "_" + "{:.2f}".format(V) + "_psi_" + str(chi) + ".pkl", 'rb') as f:                                                                                
    psi = pickle.load(f)
## ************************************************


## ********** Entropy and correlation length *****
Ent = np.max(psi.entanglement_entropy())
Cl = psi.correlation_length()
# SAVE to file if needed
## ************************************************


## ********** Spin Correlation *******************

L = 200 # the limit on the distance R
LAct = 10 # size of the iMPS unit cell
NL = L//LAct

NumA = np.real(psi.expectation_value("NA"))
NumB = np.real(psi.expectation_value("NB"))
ND = psi.sites[0].NA - psi.sites[0].NB
CorrSpin = np.zeros(L)

NumDCorr = np.real(psi.correlation_function(ND, ND, [0], np.arange(L), opstr="Id"))
NumDLoc = np.tile(NumA-NumB, NL)

CorrSpin = (NumDCorr[0]- NumDLoc * NumDLoc[0])/4

# SAVE to file if needed

## ************************************************


## ********** Charge Correlation *******************

L = 200 # the limit on the distance R
LAct = 10 # size of the iMPS unit cell
NL = L//LAct

NumA = np.real(psi.expectation_value("NA"))
NumB = np.real(psi.expectation_value("NB"))
NT = psi.sites[0].NA + psi.sites[0].NB
CorrCharge = np.zeros(L)

NumTCorr = np.real(psi.correlation_function(NT, NT, [0], np.arange(L), opstr="Id"))
NumTLoc = np.tile(NumA+NumB, NL)

CorrCharge = (NumTCorr[0]- NumTLoc * NumTLoc[0])

# SAVE to file if needed

## ************************************************



## ********** n+ Correlation **********************

L = 200 # the limit on the distance R
LAct = 10 # size of the iMPS unit cell
NL = L//LAct

NumA = np.real(psi.expectation_value("NA"))
NA = psi.sites[0].NA
CorrNplus = np.zeros(L)

NumPCorr = np.real(psi.correlation_function(NA, NA, [0], np.arange(L), opstr="Id"))
NumALoc = np.tile(NumA, NL)

CorrNPlus = (NumPCorr[0]- NumALoc * NumALoc[0])

# SAVE to file if needed

## ************************************************

