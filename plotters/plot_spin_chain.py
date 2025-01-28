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

def plot_correlations(correlations, corr_graph, title_prefix=''):
    """
    Plot correlation functions either as a 2D heatmap or 1D plot depending on the
    correlation type.

    Parameters
    ----------
    correlations : ndarray
        Correlation values array from compute_correlation
    corr_graph : LatticeGraph
        Graph object that defined the correlation function
    title_prefix : str, optional
        Optional prefix for the plot title
    """
    # Determine if this is a 1D correlation by checking if all terms are single-site
    is_1d = all(len(term) == 2 for terms in corr_graph.interaction_dict.values()
                for term in terms)

    # Get operator types from correlation graph
    op_types = list(corr_graph.interaction_dict.keys())
    op_str = ' + '.join(op_types).upper()

    if is_1d:
        # 1D plot for single-site correlations
        diag_vals = np.diag(correlations)
        sites = np.arange(len(diag_vals))

        plt.figure(figsize=(8, 6))
        plt.plot(sites, diag_vals, 'o-', markersize=8)
        plt.xlabel('Site')
        plt.ylabel(f'<{op_str}>')
        plt.grid(True)
        plt.title(f'{title_prefix}Single-Site {op_str} Correlations')

    else:
        # 2D heatmap for two-site correlations
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        # Plot heatmap
        im1 = ax1.imshow(correlations, cmap='RdBu', aspect='equal',
                        vmin=-np.abs(correlations).max(),
                        vmax=np.abs(correlations).max())
        plt.colorbar(im1, ax=ax1)
        ax1.set_xlabel('Site j')
        ax1.set_ylabel('Site i')
        ax1.set_title(f'Two-Site {op_str} Correlations')

        # Plot decay of correlations from middle site
        mid_site = len(correlations) // 2
        distances = np.arange(len(correlations) - mid_site)
        corr_decay = correlations[mid_site, mid_site:]

        ax2.plot(distances, corr_decay, 'o-', markersize=8)
        ax2.set_xlabel('Distance from middle site')
        ax2.set_ylabel(f'<{op_str}>')
        ax2.grid(True)
        ax2.set_title(f'Correlation Decay from Site {mid_site}')

        plt.suptitle(f'{title_prefix}Correlation Functions')

    plt.tight_layout()
    plt.show()

def plot_Sperp_vs_params(n_points=6, Jxy_range=(-3, 3), q_range=(-3, 3)):
    """
    Plot the expectation value of S_perp^2 as a function of Jxy and q parameters.

    Parameters
    ----------
    n_points : int, optional
        Number of points to evaluate for each parameter. Default is 6.
    Jxy_range : tuple(float, float), optional
        Range of Jxy values to scan over (min, max). Default is (-3, 3).
    q_range : tuple(float, float), optional
        Range of q values to scan over (min, max). Default is (-3, 3).
    """
    L = 10  # chain length
    bond_dims = np.asarray([2, 4, 64])

    # Create parameter grids
    Jxy_vals = np.linspace(Jxy_range[0], Jxy_range[1], n_points)
    q_vals = np.linspace(q_range[0], q_range[1], n_points)

    # Define S_perp^2 operator
    S_perp_sq_terms = [['xx', 1.0, 0], ['yy', 1.0, 0]]
    S_perp_sq_graph = LatticeGraph.from_interactions(L, S_perp_sq_terms, pbc=False)

    # Initialize results array
    S_perp_sq_nn_q_array = np.zeros((n_points, n_points))

    # Scan over parameters
    for i, Jxy in enumerate(Jxy_vals):
        for j, q in enumerate(q_vals):
            # Create XXZ2 model terms with nearest-neighbor interactions
            terms = [
                ['xx', Jxy, 'nn'],
                ['yy', Jxy, 'nn'],
                ['zz', q, np.inf]
            ]

            # Create lattice graph and DMRG engine
            graph = LatticeGraph.from_interactions(L, terms, pbc=False)
            dmrg = DMRGEngine(graph, spin='1')

            # Compute ground state
            dmrg.compute_energies_mps(bond_dims=[bond_dims[-1]], n_roots=2)

            # Compute normalized S_perp^2 expectation value
            Sperp_expectation = dmrg.compute_expectation(S_perp_sq_graph)
            Sperp_expectation /= (L*(L+1))
            S_perp_sq_nn_q_array[i, j] = np.abs(Sperp_expectation)

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot with proper extent to align grid with parameter values
    im = ax.imshow(S_perp_sq_nn_q_array,
                  extent=[q_range[0], q_range[1], Jxy_range[0], Jxy_range[1]],
                  origin='lower', aspect='auto')

    # Add labels and colorbar
    ax.set_title(r'$XXZ^2$ DMRG NN $\langle S_{\perp}^2 \rangle$')
    ax.set_ylabel(r'$J_{xy}$')
    ax.set_xlabel(r'$q$')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\langle S_{\perp}^2 \rangle/(N_{spin}(N_{spin}+1))$')

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
        ['zz', 1.5, 3]
    ]

    # Create lattice graph and DMRG engine
    graph = LatticeGraph.from_interactions(L, terms, pbc=False)
    dmrg = DMRGEngine(graph, spin='1')

    # # Run convergence test
    # energies = dmrg.run_convergence_test(bond_dims, n_roots)

    # # Plot convergence results
    # plot_convergence_results(energies, bond_dims, 'Power Law XXZ (alpha=3)', L)

    # energy, _ = dmrg.compute_energies_mps(bond_dims=[bond_dims[-1]], n_roots=n_roots)

    # # Example 1: Two-site ZZ correlations
    # zz_terms = [['zz', 1.0, 3]]
    # zz_graph = LatticeGraph.from_interactions(L, zz_terms, pbc=False)
    # zz_correlations = dmrg.compute_correlation(zz_graph)
    # plot_correlations(zz_correlations, zz_graph, 'Power Law XXZ - ')

    # # Example 2: Single-site Z correlations
    # z_terms = [['z', 1.0, np.inf]]
    # z_graph = LatticeGraph.from_interactions(L, z_terms, pbc=False)
    # z_correlations = dmrg.compute_correlation(z_graph)
    # plot_correlations(z_correlations, z_graph, 'Power Law XXZ - ')

    plot_Sperp_vs_params()
