import numpy as np
from abc import ABC, abstractmethod
from typing import Any
import quspin
from scipy.linalg import expm
from quspin.basis import spin_basis_1d
from quspin.tools.Floquet import Floquet
from quspin.tools.misc import matvec
from models.utility import HiddenPrints
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes

class LatticeGraph:
    def __init__(self, num_sites: int = None, interaction_dict: dict = None):
        """
        Initialize LatticeGraph.

        Parameters
        ----------
        num_sites : int, optional
            Number of sites in the lattice. Defaults to None.
        interaction_dict : dict, optional
            Dictionary specifying the interactions between sites. The keys are
            the interaction types, and the values are lists of lists. The
            inner lists should contain the site indices and the interaction
            strength. Defaults to an empty dictionary. See the class method
            `from_interactions` for more details.

        Attributes
        ----------
        num_sites : int
            Number of sites in the lattice.
        interaction_dict : dict
            Dictionary specifying the interactions between sites.
        """
        self.num_sites = num_sites  # number of sites
        self.interaction_dict: dict[str, list[list[Any]]] = (interaction_dict
                                                             or {})

    def __call__(self, t):
        """
        Evaluate the interaction strengths at time t.

        Parameters
        ----------
        t : float
            Time at which to evaluate the interaction strengths.

        Returns
        -------
        interaction_dict : dict
            Dictionary specifying the interaction strengths at time t. The
            keys are the interaction types, and the values are lists of lists.
            The inner lists contain the site indices and the interaction
            strength at time t.
        """
        return {op: [[f(t) if callable(f) else f for f in interaction] for
                     interaction in interactions] for op, interactions in
                self.interaction_dict.items()}

    @classmethod
    def from_interactions(cls, num_sites: int, terms: list[list[Any]],
                          pbc: bool = False):
        """
        Construct a LatticeGraph object from a list of interaction terms.

        Parameters
        ----------
        num_sites : int
            Number of sites in the lattice.
        terms : list[list[Any]]
            List of interaction terms. Each term is a list of length 3, where
            the first element is the operator (a string, e.g. 'XX', 'yy', 'z'),
            the second element is the strength of the interaction (either a
            number or a callable), and the third element is the range of the
            interaction (either a number or a string, see below).

            If the range is a string, it must be either 'nn' for nearest
            neighbor interactions, 'nnn' for next-nearest neighbor interactions,
            or 'inf' for on-site interactions.

            If the range is a number, it is the inverse range of the
            interaction. For example, alpha=3 would mean that the interaction
            strength decays as 1/r^3, where r is the distance between the two
            sites.

        pbc : bool, optional
            Whether to use periodic boundary conditions. Default is False.

        Returns
        -------
        LatticeGraph
            A new LatticeGraph object constructed from the given interaction
            terms.

        Raises
        ------
        ValueError
            If the interaction operator is not a string, or if the range is not
            a valid string or number.
        """

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
                        graph = [[lambda t, s=strength, i=i:
                                  s(t, i, (i + 1) % num_sites), i,
                                  (i + 1) % num_sites] for i
                                 in range(num_sites - 1 + int(pbc))]
                    else:
                        graph = [[strength, i, (i + 1) % num_sites]
                                 for i in range(num_sites - 1 + int(pbc))]

                # Next-Nearest Neighbor (NNN) interactions
                elif alpha == 'nnn':
                    if callable(strength):
                        graph = [[lambda t, s=strength, i=i:
                                  s(t, i, (i + 2) % num_sites), i,
                                  (i + 2) % num_sites]
                                 for i in range(num_sites - 2 + 2*int(pbc))]
                    else:
                        graph = [[strength, i, (i + 2) % num_sites]
                                 for i in range(num_sites - 2 + 2*int(pbc))]

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
                             for i in range(num_sites)]
                else:
                    # Constant on-site term
                    graph = [[strength, i]
                             for i in range(num_sites)]

                # Add to the dictionary for this operator
                new_interaction_dict[operator].extend(graph)

            else:
                if len(operator) != 2:
                    raise ValueError(f"Two-site operation requires two-site "
                                     f"operator: {operator}")

                # General long-range interaction
                if callable(strength):
                    # Time and site-dependent strength, inverse range alpha
                    graph = [[lambda t, s=strength, i=i, j=j:
                              s(t, i, j)*np.abs(j - i).astype(float)**(-1*alpha), i, j]
                             for i in range(num_sites) for j in
                             range(num_sites) if j>i]
                else:
                    # Constant interaction strength, inverse range alpha
                    graph = [[strength*np.abs(j - i).astype(float)**(-1*alpha), i, j]
                             for i in range(num_sites) for j in
                             range(num_sites) if j>i]

                # Add to the dictionary for this operator
                new_interaction_dict[operator].extend(graph)

        # Create a new graph with the constructed interaction dictionary
        return cls(num_sites, new_interaction_dict)


class ComputationStrategy(ABC):
    def __init__(self, graph: LatticeGraph, spin='1/2',
                 unit_cell_length: int = 1):
        """
        Initialize the ComputationStrategy with a graph and optional parameters.

        Parameters
        ----------
        graph : LatticeGraph
            The graph containing the interaction terms and lattice structure.
        spin : str, optional
            The spin of the particles in the lattice, defaults to '1/2'.
        unit_cell_length : int, optional
            The number of lattice sites in the unit cell, defaults to 1.
        """
        self.graph = graph
        self.spin = spin
        self.unit_cell_length = unit_cell_length

    def frobenius_loss(self, matrix1: list[list[Any]],
                       matrix2: list[list[Any]]):
        """
        Compute the fidelity metric between two Hamiltonians using the
        (normalized) Frobenius norm.

        Parameters
        ----------
        matrix1 : list[list[Any]]
            The first Hamiltonian matrix.
        matrix2 : list[list[Any]]
            The second Hamiltonian matrix.

        Returns
        -------
        float
            The fidelity metric between the two Hamiltonians. If the two input
            matrices are identical, the output will be 1.

        Notes
        -----
        This method uses the (normalized) Frobenius norm to compute a fidelity
        metric for two Hamiltonians. The Frobenius norm is defined as the square
        root of the sum of the absolute squares of the elements of a matrix. The
        normalized Frobenius norm is the Frobenius norm divided by the square root
        of the product of the Frobenius norms of the two matrices. The output of
        this method is 1 minus the normalized Frobenius norm of the difference
        between the two matrices.
        """
        overlap = self.frobenius_norm(matrix1, matrix2)
        norm = np.sqrt(self.frobenius_norm(matrix1, matrix1) *
                       self.frobenius_norm(matrix2, matrix2))
        return 1 - overlap / norm

    def frobenius_norm(self, matrix1: list[list[Any]],
                       matrix2: list[list[Any]]):
        """
        Compute the Frobenius norm of the overlap between two matrices.

        Parameters
        ----------
        matrix1 : list[list[Any]]
            The first matrix.
        matrix2 : list[list[Any]]
            The second matrix.

        Returns
        -------
        float
            The Frobenius norm of the overlap between the two matrices.
        """
        conj_matrix1 = np.matrix(matrix1).getH()
        product = matvec(conj_matrix1, matrix2)
        overlap = np.abs(np.trace(product))
        return np.sqrt(overlap)

    def norm_identity_loss(self, matrix1: list[list[Any]],
                           matrix2: list[list[Any]]):
        """
        Use the norm difference between the product of two unitaries and the
        identity to establish a loss metric for the two unitaries. Identical
        unitaries return values close to zero.

        Parameters
        ----------
        matrix1 : list[list[Any]]
            The first matrix, representing a Hamiltonian times evolution time.
        matrix2 : list[list[Any]]
            The second matrix, representing a Hamiltonian times evolution time.

        Returns
        -------
        float
            The norm difference between the product of the two unitaries and the
            identity.
        """
        unitary1 = expm(-1j * matrix1)
        unitary2 = expm(-1j * matrix2)
        conj_unitary1 = np.matrix(unitary1).getH()
        product = matvec(conj_unitary1, unitary2)
        identity = np.identity(product.shape[0])
        diff = product - identity
        conj_diff = np.matrix(diff).getH()
        norm = np.sqrt(np.abs(np.trace(matvec(conj_diff, diff))))
        return norm


class DiagonEngine(ComputationStrategy):
    def get_quspin_hamiltonian(self, t: float):
        """
        Construct the Hamiltonian for the spin chain using QuSpin.

        This method generates the Hamiltonian for the spin chain system
        at a given time `t`, formatted for use with the QuSpin library.
        It utilizes the spin_basis_1d to declare the basis of states
        and constructs the Hamiltonian using the interaction terms
        provided by the graph at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the Hamiltonian.

        Returns
        -------
        quspin.operators.hamiltonian
            The Hamiltonian object in QuSpin format for the current
            spin chain configuration.
        """
        # Declare a basis of states for the spin chain
        self.basis = spin_basis_1d(L=self.graph.num_sites,
                                   a=self.unit_cell_length,
                                   S=self.spin)
        # Put our Hamiltonian into QuSpin format
        static = [[key, self.graph(t)[key]] for key in self.graph(t).keys()]
        # Create QuSpin Hamiltonian, suppressing annoying print statements
        # TODO: is this multiplying my operators by 2?
        with HiddenPrints():
            H = quspin.operators.hamiltonian(static, [], basis=self.basis)

        return H

    def get_quspin_floquet_hamiltonian(self, params: list[float or str],
                                       dt_list: list[float]):
        """
        Construct a Floquet Hamiltonian using QuSpin for the current graph.

        Parameters
        ----------
        params : list[float or str]
            List of times or parameters to evaluate the Hamiltonian at.
        dt_list : list[float]
            Durations of each time step in the Floquet period. Set dt=0 for a
            delta pulse (make sure you've integrated the Hamiltonian to
            accrue the proper amount of phase)

        Returns
        -------
        quspin.operators.floquet
            The Floquet Hamiltonian constructed using the given parameters.
        """
        if len(params) != len(dt_list):
            raise ValueError("paramList and dtList must have the same length")

        H_list = [self.get_quspin_hamiltonian(t) for t in params]
        floquet_period = sum(dt_list)
        # if there are elements value 0 in dtList (indicating a delta pulse),
        # replace them with 1 for QuSpin's integrator
        dt_list = [dt if dt > 0 else 1 for dt in dt_list]
        evo_dict = {"H_list": H_list, "dt_list": dt_list, "T": floquet_period}
        # the Hamiltonian could also be computed on-the-fly by QuSpin, but for
        # sake of clarity and speed, we'll generate the list of Hamiltonians
        # ahead of time
        results = Floquet(evo_dict, HF=True, UF=False, force_ONB=True, n_jobs=1)

        return results.HF

    def frobenius_loss(self, matrix1, matrix2):
        """
        Compute the Frobenius loss between two matrices.

        Parameters
        ----------
        matrix1 : quspin.operators.hamiltonian or np.ndarray
            The first matrix.
        matrix2 : quspin.operators.hamiltonian or np.ndarray
            The second matrix.

        Returns
        -------
        float
            The Frobenius loss between the two matrices.
        """
        if isinstance(matrix1, quspin.operators.hamiltonian):
            matrix1 = matrix1.todense()
        if isinstance(matrix2, quspin.operators.hamiltonian):
            matrix2 = matrix2.todense()

        return super().frobenius_loss(matrix1, matrix2)

    def frobenius_norm(self, matrix1, matrix2):
        """
        Compute the Frobenius norm between two matrices.

        Parameters
        ----------
        matrix1 : quspin.operators.hamiltonian or np.ndarray
            The first matrix.
        matrix2 : quspin.operators.hamiltonian or np.ndarray
            The second matrix.

        Returns
        -------
        float
            The Frobenius norm between the two matrices.
        """
        if isinstance(matrix1, quspin.operators.hamiltonian):
            matrix1 = matrix1.todense()
        if isinstance(matrix2, quspin.operators.hamiltonian):
            matrix2 = matrix2.todense()

        return super().frobenius_norm(matrix1, matrix2)

    def norm_identity_loss(self, matrix1, matrix2):
        """
        Compute the norm identity loss between two matrices.

        This method calculates a fidelity metric between two matrices using a
        norm identity approach. It converts the matrices to dense format if they
        are instances of `quspin.operators.hamiltonian` before computing the loss.

        Parameters
        ----------
        matrix1 : quspin.operators.hamiltonian or np.ndarray
            The first matrix.
        matrix2 : quspin.operators.hamiltonian or np.ndarray
            The second matrix.

        Returns
        -------
        float
            The norm identity loss between the two matrices.
        """
        if isinstance(matrix1, quspin.operators.hamiltonian):
            matrix1 = matrix1.todense()
        if isinstance(matrix2, quspin.operators.hamiltonian):
            matrix2 = matrix2.todense()

        return super().norm_identity_loss(matrix1, matrix2)


class DMRGEngine(ComputationStrategy):
    def __init__(self, graph: LatticeGraph, spin='1', unit_cell_length: int = 1):
        """
        Initialize the DMRG Engine for solving spin chain problems.

        Parameters
        ----------
        graph : LatticeGraph
            The lattice graph defining the spin chain system.
        spin : str, optional
            The spin representation to use. Default is '1' for spin-1.
        unit_cell_length : int, optional
            Length of the unit cell. Default is 1.
        """
        super().__init__(graph, spin, unit_cell_length)
        self.energies = None  # List to store computed energies
        self.states = None    # List to store computed states
        self.driver = self._initialize_driver()

    def _initialize_driver(self):
        """
        Initialize the DMRG driver with appropriate symmetry and system parameters.

        Returns
        -------
        DMRGDriver
            Initialized DMRG driver instance.
        """
        # Initialize DMRG driver
        driver = DMRGDriver(scratch="./dmrg_tmp", symm_type=SymmetryTypes.SGB)

        # Initialize system based on spin type
        heis_twos = int(2*float(eval(self.spin)))
        driver.initialize_system(n_sites=self.graph.num_sites, heis_twos=heis_twos, heis_twosz=0)

        return driver

    def _build_mpo(self, t: float = 0.0):
        """
        Build the MPO (Matrix Product Operator) for the Hamiltonian.

        Parameters
        ----------
        t : float, optional
            Time at which to evaluate the Hamiltonian. Default is 0.0.

        Returns
        -------
        MPO
            The constructed Matrix Product Operator representing the Hamiltonian.
        """
        # Get interaction terms from the graph at time t
        interactions = self.graph(t)

        # Build Hamiltonian expression
        b = self.driver.expr_builder()

        # Process all interactions from the graph
        for op_type, terms in interactions.items():
            for term in terms:
                if len(term) < 2:  # Skip invalid terms
                    continue

                J = term[0]  # Interaction strength

                if len(term) == 2:  # Single-site term
                    site = term[1]
                    if op_type.lower() == 'z':
                        b.add_term("Z", [site], J)
                    elif op_type.lower() == 'x':
                        b.add_term("X", [site], J)
                    elif op_type.lower() == 'y':
                        b.add_term("Y", [site], J)

                elif len(term) == 3:  # Two-site term
                    site1, site2 = term[1], term[2]
                    # TODO: add appropriate check here to change xx+yy to pm+mp
                    if op_type.lower() == 'xx':
                        # XX terms are implemented as (P+M- + M+P-)/2
                        b.add_term("PM", [site1, site2], 0.5 * J)
                        b.add_term("MP", [site1, site2], 0.5 * J)
                    elif op_type.lower() == 'yy':
                        # YY terms are implemented as -(P+M- + M+P-)/2
                        b.add_term("PM", [site1, site2], -0.5 * J)
                        b.add_term("MP", [site1, site2], -0.5 * J)
                    elif op_type.lower() == 'zz':
                        b.add_term("ZZ", [site1, site2], J)

        # Define and return the MPO
        return self.driver.get_mpo(b.finalize(adjust_order=True), algo_type=MPOAlgorithmTypes.FastBipartite)

    def compute_energies_mps(self, bond_dims=[50, 100, 200], n_roots=1, t: float = 0.0):
        """
        Compute energies and MPS states using DMRG.

        Parameters
        ----------
        bond_dims : list, optional
            List of bond dimensions to use in DMRG. For example, [50, 100, 200] will run
            DMRG first with bond dimension 50, then 100, then 200, to converge the calculation
            to higher degrees of accuracy in the state correlations.
        n_roots : int, optional
            Number of states to compute. Default is 1.
        t : float, optional
            Time at which to evaluate the Hamiltonian. Default is 0.0.

        Returns
        -------
        tuple
            (energies, states) where energies is a list of computed energies
            and states is a list of computed MPS states.
        """
        # Build the MPO
        heis_mpo = self._build_mpo(t)

        energies = []
        states = []

        # Get initial random MPS
        ket = self.driver.get_random_mps(tag="KET", bond_dim=min(10, bond_dims[0]), nroots=n_roots)

        # Set up DMRG parameters
        noises = [1e-3] * 5 + [1e-5] * 5 + [0]
        thrds = [1e-8] * 10

        # Run DMRG
        energy = self.driver.dmrg(
            heis_mpo,
            ket,
            n_sweeps=100,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            cutoff=1E-20,
            tol=1e-6
        )

        # Save results
        energies.append(energy)
        mps = self.driver.load_mps(tag="KET", nroots=n_roots)

        mps = [self.driver.split_mps(mps, iroot=i, tag=f"KET_state_{i}")
                    for i in range(n_roots)]

        states.append(mps)

        self.energies = energies
        self.states = states
        return energies, states

    def compute_correlation(self, correlation_graph: LatticeGraph, state_idx=0):
        """
        Compute correlation functions for a given state using a LatticeGraph object
        that defines the correlation function to compute.

        Parameters
        ----------
        correlation_graph : LatticeGraph
            Graph object defining the correlation function. The interaction_dict should
            contain operator terms (e.g., 'x', 'y', 'z', 'xx', 'yy', 'zz') with their
            corresponding site indices and strengths. For example:
            {'z': [[1.0, i, j]] for correlator <Z_i Z_j>
            {'xx': [[1.0, i, j]], 'yy': [[1.0, i, j]]} for correlator <X_i X_j + Y_i Y_j>
        state_idx : int, optional
            Index of the state to compute correlations for. Default is 0.

        Returns
        -------
        numpy.ndarray
            Correlation values.

        Raises
        ------
        ValueError
            If no states are available or if correlation_graph has invalid operators.
        """
        if not self.states:
            raise ValueError("No states available. Run calculation first.")

        if correlation_graph.num_sites != self.graph.num_sites:
            raise ValueError("Correlation graph must have same number of sites as the system")

        # Initialize driver
        driver = DMRGDriver(scratch="./dmrg_tmp", symm_type=SymmetryTypes.SGB)
        heis_twos = int(2*float(eval(self.spin)))
        driver.initialize_system(n_sites=self.graph.num_sites, heis_twos=heis_twos, heis_twosz=0)

        # Get the state
        mps = self.states[0][state_idx]

        # Initialize correlation array
        N = self.graph.num_sites
        correlations = np.empty((N, N))

        # Operator mapping from lowercase to DMRG convention
        op_map = {'x': 'X', 'y': 'Y', 'z': 'Z'}

        # Process each operator type and its terms
        for op_type, terms in correlation_graph.interaction_dict.items():
            op_type = op_type.lower()
            if len(op_type) == 1:
                op = op_map.get(op_type)
            elif len(op_type) == 2:
                op = op_map.get(op_type[0]) + op_map.get(op_type[1])
            else:
                raise ValueError(f"Invalid operator type: {op_type}")

            if op is None:
                raise ValueError(f"Invalid operator type: {op_type}")

            # Compute expectation value for each term
            for term in terms:
                b = driver.expr_builder()
                strength = term[0] if callable(term[0]) else float(term[0])

                if len(term) == 2:  # Single-site operator
                    print(op, term[1], strength)
                    b.add_term(op, [term[1]], strength)
                    mpo = driver.get_mpo(b.finalize())
                    correlations[term[1], term[1]] = driver.expectation(mps, mpo, mps)
                    print(correlations[term[1], term[1]])
                elif len(term) == 3:  # Two-site operator
                    print(op, term[1], term[2], strength)
                    b.add_term(op, [term[1], term[2]], strength)
                    mpo = driver.get_mpo(b.finalize())
                    val = driver.expectation(mps, mpo, mps)
                    print(val)
                    correlations[term[1], term[2]] = val
                    correlations[term[2], term[1]] = val  # Symmetrize

        return correlations

    def run_convergence_test(self, bond_dims, n_roots=2):
        """
        Run convergence test over different bond dimensions.

        Parameters
        ----------
        bond_dims : array-like
            List of bond dimensions to test
        n_roots : int
            Number of excited states to compute

        Returns
        -------
        ndarray
            Array of shape (len(bond_dims), n_roots) containing energies for each bond dimension
            and excited state
        """
        energies = []

        for chi in bond_dims:
            print(f"Testing chi = {chi}")
            energy, _ = self.compute_energies_mps(bond_dims=[chi], n_roots=n_roots)
            energies.append(energy)

        return np.asarray(energies)


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
    graph = LatticeGraph.from_interactions(4, terms, pbc=True)

    print(graph("-DM"))

    computation = DiagonEngine(graph)
