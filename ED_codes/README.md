# Exact diagonalization codes

Example codes to compute the exact diagonalization of the system.

=================================

ED_lib.py --> contains the basic functions to create the occupation basis and single and two-body operators, given the filling

hamiltonian_terms.py --> ready-to-run functions to generate the operators that appear in the hamiltonian (hopping term, rung interaction, Rydberg dressing interaction for arbitrary range) in a sparse format

fidelity.py --> function to compute the fidelity between ground states in two different points of the phase diagram

tutorial.py --> small tutorial that shows how to use all the basic functions uploaded. it can be directly use to reproduce the reported results.
