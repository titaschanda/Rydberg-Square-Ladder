# iMDRG Codes based on TeNPy 
=================================


helper_func.py --> Basic function to generate local operators
model.py --> Implementation of Bosons with spins (two leg ladder)
run.py --> generate iMPS states and save on disk (input parameters: chi (bond dimension), U, and V) 

Usage:
  python run.py chi U V
  
 calculate_obs.py --> Calculate observables used in the paper from the stored iMPSs
 
Usage:
  python calculate_obs.py chi U V
