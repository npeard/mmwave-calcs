import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Any, Dict
import quspin
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.tools.Floquet import Floquet
from quspin.tools.misc import matvec


class LatticeGraph:
    def __init__(self, L: int = None, interaction_dict: dict = None):
        self.L = L  # number of sites
        self.interaction_dict: Dict[str, List[List[Any]]] = interaction_dict or {}

    def __call__(self, t):
        return {op: [[f(t) if callable(f) else f for f in interaction] for
                     interaction in interactions] for op, interactions in
                self.interaction_dict.items()}

    @classmethod
    def from_interactions(cls, L: int, terms: List[List[Any]]):
        # Create a fresh interaction dictionary
        new_interaction_dict = {}

        for term in terms:
            # Unpack term: operator, strength, interaction range
            operator, strength, alpha = term

            if not isinstance(operator, str):
                raise ValueError(f"Invalid interaction operator, "
                                 f"expected string: {operator}")

            # Normalize operator to lowercase for consistency
            operator = operator.lower()

            # Initialize list for this operator if not exists
            if operator not in new_interaction_dict:
                new_interaction_dict[operator] = []

            if isinstance(alpha, str):
                if len(operator) != 2:
                    raise ValueError(f"Two-site operation requires two-site "
                                     f"operator: {operator}")

                # Nearest Neighbor (NN) interactions
                if alpha == 'nn':
                    if callable(strength):
                        graph = [[lambda t, s=strength, i=i: s(t, i, i + 1), i,
                                  i + 1]
                                 for i in range(L - 1)]
                    else:
                        graph = [[lambda t, s=strength: s, i, i + 1]
                                 for i in range(L - 1)]

                # Next-Nearest Neighbor (NNN) interactions
                elif alpha == 'nnn':
                    if callable(strength):
                        graph = [[lambda t, s=strength, i=i: s(t, i, i + 2), i,
                                  i + 2]
                                 for i in range(L - 2)]
                    else:
                        graph = [[lambda t, s=strength: s, i, i + 2]
                                 for i in range(L - 2)]

                else:
                    raise ValueError(f"Invalid range cutoff string: {alpha}")

                # Add to the dictionary for this operator
                new_interaction_dict[operator].extend(graph)

            elif alpha == np.inf:
                if len(operator) != 1:
                    raise ValueError(f"One-site operation requires one-site "
                                     f"operator: {operator}")

                # On-site interaction
                if callable(strength):
                    # Time-dependent or site-specific on-site term
                    graph = [[lambda t, s=strength, i=i: s(t, i), i]
                             for i in range(L)]
                else:
                    # Constant on-site term
                    graph = [[lambda t, s=strength: s, i]
                             for i in range(L)]

                # Add to the dictionary for this operator
                new_interaction_dict[operator].extend(graph)

            else:
                if len(operator) != 2:
                    raise ValueError(f"Two-site operation requires two-site "
                                     f"operator: {operator}")

                # General long-range interaction
                if callable(strength):
                    # Time and site-dependent interaction
                    graph = [[lambda t, s=strength, i=i, j=j: s(t, i, j), i, j]
                             for i in range(L) for j in range(L)]
                else:
                    # Constant interaction strength
                    graph = [[lambda t, s=strength: s, i, j]
                             for i in range(L) for j in range(L)]

                # Add to the dictionary for this operator
                new_interaction_dict[operator].extend(graph)

        # Create a new Hamiltonian with the constructed interaction dictionary
        return cls(L, new_interaction_dict)


class ComputationStrategy(ABC):
    def __init__(self, graph: LatticeGraph, spin='1/2'):
        self.graph = graph
        self.spin = spin

    @abstractmethod
    def run_calculation(self, t: float = 0.0):
        """
        Run specific computational method

        Parameters:
        -----------
        t : float, optional
            Time point for Hamiltonian construction
        """
        pass

    def hilbert_schmidt_fidelity(self, H1, H2):
        # Use the Hilbert-Schmidt inner product to compute a fidelity metric for two
        # Hamiltonians. If the two input matrices are identical, the output will be
        # 1.
        conjH1 = np.matrix(H1).getH()
        conjH2 = np.matrix(H2).getH()
        product = matvec(conjH1, H2)
        overlap = np.abs(np.trace(product))
        norm = np.sqrt(np.abs(np.trace(matvec(conjH1, H1))) * np.abs(
            np.trace(matvec(conjH2, H2))))
        return overlap / norm

    def norm_identity_loss(self, H1, H2):
        # Use the norm difference between the product of two unitaries and the
        # identity to establish a loss metric for the two unitaries. Identical
        # unitaries return values close to zero.
        conjH1 = np.matrix(H1).getH()
        conjH2 = np.matrix(H2).getH()
        product = matvec(conjH1, H2)
        identity = np.identity(product.shape[0])
        diff = product - identity
        conjdiff = np.matrix(diff).getH()
        norm = np.sqrt(np.abs(np.trace(matvec(conjdiff, diff))))
        return norm


class DiagonEngine(ComputationStrategy):
    def get_quspin_hamiltonian(self, t: float, a: int = 1):
        self.basis = spin_basis_1d(L=self.graph.L, a=a, S=self.spin)
        # Put our Hamiltonian into QuSpin format
        static = [[key, self.graph(t)[key]] for key in self.graph(t).keys()]
        # Create QuSpin Hamiltonian
        # TODO: is this multiplying my operators by 2?
        H = quspin.operators.hamiltonian(static, [], basis=self.basis)

        return H

    def run_calculation(self, t: float = 0.0):
        pass

    def get_quspin_floquet_hamiltonian(self, params: List[float],
                                       dt_list: List[float]):
        # paramList could be a list of times to evaluate the Hamiltonian
        # but really it is just a list of parameters because the time dimension
        # is implicit from dtList. Set dt=0 for a delta pulse (make sure
        # you've integrated the Hamiltonian to accrue the proper amount of
        # phase)
        if len(params) != len(dt_list):
            raise ValueError("paramList and dtList must have the same length")

        H_list = [self.get_quspin_hamiltonian(t) for t in params]
        T = sum(dt_list)
        # if there are elements value 0 in dtList (indicating a delta pulse),
        # replace them with 1 for QuSpin's integrator
        dt_list = [dt if dt > 0 else 1 for dt in dt_list]
        evo_dict = {"H_list": H_list, "dt_list": dt_list, "T": T}
        # the Hamiltonian could also be computed on-the-fly by QuSpin, but for
        # sake of clarity and speed, we'll generate the list of Hamiltonians
        # ahead of time
        results = Floquet(evo_dict, HF=True, UF=True)

        return results.HF, results.UF

    def hilbert_schmidt_fidelity(self, H1, H2):
        # Override for formatting
        if isinstance(H1, quspin.operators.hamiltonian):
            H1 = H1.todense()
        if isinstance(H2, quspin.operators.hamiltonian):
            H2 = H2.todense()

        return super().hilbert_schmidt_fidelity(H1, H2)

    def norm_identity_loss(self, H1, H2):
        # Override for formatting
        if isinstance(H1, quspin.operators.hamiltonian):
            H1 = H1.todense()
        if isinstance(H2, quspin.operators.hamiltonian):
            H2 = H2.todense()

        return super().norm_identity_loss(H1, H2)


class DMRGEngine(ComputationStrategy):
    def run_calculation(self, t: float = 0.0):
        pass


if __name__ == "__main__":
    # Example usage
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

    terms = [['XX', native, 'nn'], ['yy', native, 'nn'],
             ['z', DM_z_period4, np.inf], ['z', XY_z_period4, np.inf]]
    graph = LatticeGraph.from_interactions(4, terms)

    print(graph("-DM"))

    computation = DiagonEngine(graph)
    # H = computation.quspin_hamiltonian("-DM", a = 4)
    # print(H)
    # computation.run_calculation(0.0)
    tD = 1
    tJ = 1
    tmJ = 1
    paramList = ["nat", "+DM", "nat", "+XY", "nat", "-XY", "nat", "-DM", "nat"]
    dtList = [tJ, 0, tD, 0, 2 * tmJ, 0, tD, 0, tJ]
    HF, UF = computation.get_quspin_floquet_hamiltonian(paramList, dtList)
    # print(HF)
    fid = computation.hilbert_schmidt_fidelity(HF,
                                               computation.get_quspin_hamiltonian(0))
    print(fid)
