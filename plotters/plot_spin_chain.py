import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from models.spin_chain import LatticeGraph, DMRGEngine

# Parameters
L = 20  # chain length
bond_dims = np.asarray([2, 4, 8, 16, 32, 64, 128])
Delta = np.asarray([-1.5, -0.5, 0.5, 1.5])
n_roots = 2  # compute ground state and first excited state

def get_XXZ_energy(alpha=None, Jxy=1, Delta=1, chi=50):
    """
    Compute the energy per spin for the XXZ model with given parameters.
    
    Parameters
    ----------
    alpha : float or None
        Power law decay of interactions. None for nearest-neighbor,
        0 for dense interactions, other values for power law decay.
    Jxy : float
        XY coupling strength
    Delta : float
        Anisotropy parameter
    chi : int
        Bond dimension for DMRG
    
    Returns
    -------
    tuple
        (energy_per_spin, mps_states)
    """
    # Define the XXZ model terms
    terms = []
    
    # XX and YY terms
    if alpha is None:
        # Nearest neighbor case
        terms.extend([
            ['xx', Jxy, 'nn'],
            ['yy', Jxy, 'nn']
        ])
    else:
        # Power law decay case
        terms.extend([
            ['xx', Jxy, alpha],
            ['yy', Jxy, alpha]
        ])
    
    # ZZ terms with same spatial dependence
    if alpha is None:
        terms.append(['zz', Jxy * Delta, 'nn'])
    else:
        terms.append(['zz', Jxy * Delta, alpha])
    
    # Create lattice graph and DMRG engine
    graph = LatticeGraph.from_interactions(L, terms, pbc=False)
    dmrg = DMRGEngine(graph, spin='1/2')
    
    # Compute energies and states
    energies, states = dmrg.compute_energies_mps(bond_dims=[chi], n_roots=n_roots)
    
    return np.array(energies), states

# Lists to store results
EperSpin_nn = []  # nearest neighbor
EperSpin_r3 = []  # power law r^-3
EperSpin_dense = []  # dense interactions

# Compute energies for different Delta values and bond dimensions
print("Computing nearest neighbor interactions...")
for delta in Delta:
    for chi in bond_dims:
        print(f"Delta = {delta}, chi = {chi}")
        energy, _ = get_XXZ_energy(alpha=None, Jxy=1, Delta=delta, chi=chi)
        EperSpin_nn.append(energy)

print("\nComputing r^-3 interactions...")
for delta in Delta:
    for chi in bond_dims:
        print(f"Delta = {delta}, chi = {chi}")
        energy, _ = get_XXZ_energy(alpha=3, Jxy=1, Delta=delta, chi=chi)
        EperSpin_r3.append(energy)

print("\nComputing dense interactions...")
for delta in Delta:
    for chi in bond_dims:
        print(f"Delta = {delta}, chi = {chi}")
        energy, _ = get_XXZ_energy(alpha=0, Jxy=1, Delta=delta, chi=chi)
        EperSpin_dense.append(energy)

# Reshape results
EperSpin_nn_array = np.asarray(EperSpin_nn).reshape((len(Delta), len(bond_dims), n_roots))
EperSpin_r3_array = np.asarray(EperSpin_r3).reshape((len(Delta), len(bond_dims), n_roots))
EperSpin_dense_array = np.asarray(EperSpin_dense).reshape((len(Delta), len(bond_dims), n_roots))

# Plot results
# Nearest neighbor case
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
for i, delta in enumerate(Delta):
    converged_gs_energy = EperSpin_nn_array[i, -1, 0]  # Use highest bond dim as reference
    converged_s1_energy = EperSpin_nn_array[i, -1, 1]
    
    ax1.plot(bond_dims, np.abs(EperSpin_nn_array[i, :, 0] - converged_gs_energy),
             marker='.', markersize=10, linestyle='solid',
             label=f'$\\Delta = {delta}$')
    ax2.plot(bond_dims, np.abs(EperSpin_nn_array[i, :, 1] - converged_s1_energy),
             marker='.', markersize=10, linestyle='dotted',
             label=f'$\\Delta = {delta}$')

ax1.set_xlabel(r'$\chi$')
ax1.set_ylabel('Energy Error')
ax1.set_yscale('log')
ax1.set_xscale('log', base=2)
ax1.grid(True)
ax1.legend()
ax1.set_title('Ground State')

ax2.set_xlabel(r'$\chi$')
ax2.set_ylabel('Energy Error')
ax2.set_yscale('log')
ax2.set_xscale('log', base=2)
ax2.grid(True)
ax2.legend()
ax2.set_title('First Excited State')

fig.suptitle(f'XXZ Energy Error vs Bond Dimension (Nearest-Neighbor), L = {L}')
plt.tight_layout()
plt.savefig('xxz_energy_error_nn.png')
plt.show()

# r^-3 case
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
for i, delta in enumerate(Delta):
    converged_gs_energy = EperSpin_r3_array[i, -1, 0]
    converged_s1_energy = EperSpin_r3_array[i, -1, 1]
    
    ax1.plot(bond_dims, np.abs(EperSpin_r3_array[i, :, 0] - converged_gs_energy),
             marker='.', markersize=10, linestyle='solid',
             label=f'$\\Delta = {delta}$')
    ax2.plot(bond_dims, np.abs(EperSpin_r3_array[i, :, 1] - converged_s1_energy),
             marker='.', markersize=10, linestyle='dotted',
             label=f'$\\Delta = {delta}$')

ax1.set_xlabel(r'$\chi$')
ax1.set_ylabel('Energy Error')
ax1.set_yscale('log')
ax1.set_xscale('log', base=2)
ax1.grid(True)
ax1.legend()
ax1.set_title('Ground State')

ax2.set_xlabel(r'$\chi$')
ax2.set_ylabel('Energy Error')
ax2.set_yscale('log')
ax2.set_xscale('log', base=2)
ax2.grid(True)
ax2.legend()
ax2.set_title('First Excited State')

fig.suptitle(f'XXZ Energy Error vs Bond Dimension (Dipolar), L = {L}')
plt.tight_layout()
plt.savefig('xxz_energy_error_r3.png')
plt.show()

# Dense case
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
for i, delta in enumerate(Delta):
    converged_gs_energy = EperSpin_dense_array[i, -1, 0]
    converged_s1_energy = EperSpin_dense_array[i, -1, 1]
    
    ax1.plot(bond_dims, np.abs(EperSpin_dense_array[i, :, 0] - converged_gs_energy),
             marker='.', markersize=10, linestyle='solid',
             label=f'$\\Delta = {delta}$')
    ax2.plot(bond_dims, np.abs(EperSpin_dense_array[i, :, 1] - converged_s1_energy),
             marker='.', markersize=10, linestyle='dotted',
             label=f'$\\Delta = {delta}$')

ax1.set_xlabel(r'$\chi$')
ax1.set_ylabel('Energy Error')
ax1.set_yscale('log')
ax1.set_xscale('log', base=2)
ax1.grid(True)
ax1.legend()
ax1.set_title('Ground State')

ax2.set_xlabel(r'$\chi$')
ax2.set_ylabel('Energy Error')
ax2.set_yscale('log')
ax2.set_xscale('log', base=2)
ax2.grid(True)
ax2.legend()
ax2.set_title('First Excited State')

fig.suptitle(f'XXZ Energy Error vs Bond Dimension (Dense), L = {L}')
plt.tight_layout()
plt.savefig('xxz_energy_error_dense.png')
plt.show()
