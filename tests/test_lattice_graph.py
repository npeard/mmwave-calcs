import sys
import os
import pytest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.spin_chain import LatticeGraph

def DM_z_period4(t, i):
    phase = np.pi / 2 * (i % 4)
    if t == "+DM":
        return phase
    elif t == "-DM":
        return -phase
    else:
        return 0


def XY_z_period4(t, i):
    phase = np.pi - 3. * np.pi / 2 * (i % 4)
    if t == "+XY":
        return phase
    elif t == "-XY":
        return -phase
    else:
        return 0


def native(t, i, j):
    if t in ["+DM", "-DM", "+XY", "-XY"]:
        return 0
    else:
        return 0.5


def test_from_interactions():
    minus_DM_dict = {'XX': [[0, 0, 1], [0, 1, 2], [0, 2, 3]],
                     'YY': [[0, 0, 1], [0, 1, 2], [0, 2, 3]],
                     'Z': [[-0.0, 0], [-1.5707963267948966, 1],
                           [-3.141592653589793, 2], [-4.71238898038469, 3],
                           [0, 0], [0, 1], [0, 2], [0, 3]]}
    minus_DM_pbc_dict = {'XX': [[0, 0, 1], [0, 1, 2], [0, 2, 3], [0, 3, 0]],
                         'YY': [[0, 0, 1], [0, 1, 2], [0, 2, 3], [0, 3, 0]],
                         'Z': [[-0.0, 0], [-1.5707963267948966, 1],
                               [-3.141592653589793, 2],
                               [-4.71238898038469, 3], [0, 0], [0, 1],
                               [0, 2], [0, 3]]}

    terms = [['XX', native, 'nn'], ['yy', native, 'nn'],
             ['z', DM_z_period4, np.inf], ['z', XY_z_period4, np.inf]]
    graph = LatticeGraph.from_interactions(4, terms, pbc=False)
    graph_pbc = LatticeGraph.from_interactions(4, terms, pbc=True)

    assert minus_DM_dict == graph("-DM")
    assert minus_DM_pbc_dict == graph_pbc("-DM")


def test_torch_lattice_graph_equivalence():
    """Test that TorchLatticeGraph and LatticeGraph produce equivalent interaction graphs."""
    from models.torch_chain import TorchLatticeGraph, TorchParameter

    # Define some test parameters
    num_sites = 4

    # Define a simple parameter for testing
    def param_func(t, *args, **kwargs):
        scale = kwargs['scale'].detach().numpy()
        return scale * t

    torch_param = TorchParameter({'scale': 1.0}, param_func)

    # Define interaction terms for both graph types
    terms = [
        ['XX', 1.0, 'nn'],  # Static nearest-neighbor XX interaction
        ['ZZ', torch_param, 'nn'],  # Time-dependent nearest-neighbor ZZ interaction
        ['Y', 0.5, np.inf]  # On-site Y field
    ]

    # Create both types of graphs
    torch_graph = TorchLatticeGraph.from_torch_interactions(num_sites, terms, pbc=False)
    regular_graph = LatticeGraph.from_interactions(num_sites, terms, pbc=False)

    # Test at different times
    test_times = [0.0, 0.5, 1.0]
    for t in test_times:
        torch_interactions = torch_graph(t)
        regular_interactions = regular_graph(t)

        # Check that both graphs have the same operators
        assert set(torch_interactions.keys()) == set(regular_interactions.keys()), \
            f"Operators don't match at t={t}"

        # For each operator, check that interactions match
        for op in torch_interactions.keys():
            torch_terms = sorted(torch_interactions[op])
            regular_terms = sorted(regular_interactions[op])

            # Compare the number of terms
            assert len(torch_terms) == len(regular_terms), \
                f"Number of {op} terms don't match at t={t}"

            # Compare each term
            for torch_term, regular_term in zip(torch_terms, regular_terms):
                # Convert any tensor values to numpy for comparison
                torch_term_np = [t.item() if hasattr(t, 'item') else t for t in torch_term]

                np.testing.assert_allclose(torch_term_np, regular_term,
                    rtol=1e-7, err_msg=f"{op} terms don't match at t={t}")
