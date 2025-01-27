import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from models.spin_chain import LatticeGraph, DMRGEngine

def plot_convergence_results(energies, bond_dims, graph_desc, L):
    """
    Plot convergence test results.

    Parameters
    ----------
    energies : ndarray
        Array of shape (len(bond_dims), n_roots) containing energies
    bond_dims : array-like
        List of bond dimensions used
    graph_desc : str
        Description of the graph being tested (e.g. 'Nearest-Neighbor XXZ')
    L : int
        System size
    """
    n_roots = energies.shape[-1]
    energies = np.squeeze(energies, axis=1)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    # Use highest bond dimension as reference for convergence
    print(energies.shape)
    converged_gs_energy = energies[-1, 0]
    converged_s1_energy = energies[-1, 1] if n_roots > 1 else None

    ax1.plot(bond_dims, np.abs(energies[:, 0]/converged_gs_energy - 1),
            marker='.', markersize=10, linestyle='solid')

    if converged_s1_energy is not None:
        ax2.plot(bond_dims, np.abs(energies[:, 1]/converged_s1_energy - 1),
                marker='.', markersize=10, linestyle='dotted')

    for ax, title in [(ax1, 'Ground State'), (ax2, 'First Excited State')]:
        ax.set_xlabel(r'$\chi$')
        ax.set_ylabel('Relative Energy Error')
        ax.set_yscale('log')
        ax.set_xscale('log', base=2)
        ax.grid(True)
        ax.set_title(title)

    fig.suptitle(f'XXZ Energy Error vs Bond Dimension ({graph_desc}), L = {L}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage
    L = 20  # chain length
    bond_dims = np.asarray([2, 4, 8, 16])
    n_roots = 2  # compute ground state and first excited state

    # Create example XXZ model terms with power law interactions (alpha=3)
    terms = [
        ['xx', 1, 3],
        ['yy', 1, 3],
        ['zz', -1.5, 3]
    ]

    # Create lattice graph and DMRG engine
    graph = LatticeGraph.from_interactions(L, terms, pbc=False)
    dmrg = DMRGEngine(graph, spin='1')

    # Run convergence test
    energies = dmrg.run_convergence_test(bond_dims, n_roots)

    # Plot results
    plot_convergence_results(energies, bond_dims, 'Power Law XXZ (alpha=3)', L)
