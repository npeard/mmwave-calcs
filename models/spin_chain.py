import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Any
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.tools.Floquet import Floquet, Floquet_t_vec  # Floquet Hamiltonian
from quspin.operators import hamiltonian  # Hamiltonians
from quspin.tools.misc import matvec
from scipy.linalg import expm


class LatticeHamiltonian:
    def __init__(self, graph_data: List[List[Any]] = None, L: int = None,):
        self.L = L # number of sites
        self.terms = None # list of terms in the Hamiltonian
        # Initialize graph data, default to empty list if not provided
        self.graph_data = graph_data or []
    
    def __call__(self, t):
        print(self.graph_data)
        return [[f(t) if callable(f) else f for f in row] for row in
                self.graph_data]
    
    def construct_graph(self):
        # Reinitialize the Hamiltonian,
        # construct_graph should typically be called once
        new_graph_data = []
        
        for term in self.terms:
            # operator is the interaction operator, e.g. 'ZZ' for two-site
            # and 'Z' for one-site
            # strength is the magnitude of the interaction,
            # alpha is the inverse range of interaction
            operator, strength, alpha = term
            
            if not isinstance(operator, str):
                raise ValueError(f"Invalid interaction operator, "
                                 f"expected string: {operator}")
            
            if isinstance(alpha, str):
                if len(operator) != 2:
                    raise ValueError(f"Two-site operation requires two-site "
                                     f"operator: {operator}")
                
                # Long-range interactions with string-based cutoff (e.g., 'NN', 'NNN')
                if alpha == 'NN':
                    if callable(strength):
                        graph = [[lambda t, i=i, j=i+1: strength(t, i, j), i,
                                  i+1, operator]
                                 for i in range(self.L-1)]
                    else:
                        graph = [[strength, i, i+1, operator] for i in range(
                            self.L-1)]
                elif alpha == 'NNN':
                    if callable(strength):
                        graph = [[lambda t, i=i, j=i+2: strength(t, i, j), i,
                                  i+2, operator]
                                for i in range(self.L-2)]
                    else:
                        graph = [[strength, i, i+2, operator] for i in range(
                            self.L-2)]
                else:
                    raise ValueError(f"Invalid range cutoff string: {alpha}")
            
            elif alpha == np.infty:
                if len(operator) != 1:
                    raise ValueError(f"One-site operation requires one-site "
                                     f"operator: {operator}")
                
                # On-site interaction
                if callable(strength):
                    # Time-dependent or site-specific on-site term
                    graph = [[lambda t, i=i: strength(t, i), i, operator] for
                             i in range(self.L)]
                else:
                    # Constant on-site term
                    graph = [[strength, i, operator] for i in range(self.L)]
                    
            else:
                if len(operator) != 2:
                    raise ValueError(f"Two-site operation requires two-site "
                                     f"operator: {operator}")
                # General long-range interaction
                if callable(strength):
                    # Time and site-dependent interaction
                    graph = [
                        [lambda t, i=i, j=j: strength(t, i, j), i, j, operator]
                        for i in range(self.L)
                        for j in range(self.L)
                    ]
                else:
                    # Constant interaction strength
                    graph = [[strength, i, j, operator] for i in range(self.L)
                             for j in range(self.L)]
            # for each term and graph generated from above conditions
            new_graph_data.extend(graph)
        
        # Return a new Hamiltonian instance with the constructed graph
        return LatticeHamiltonian(new_graph_data, self.L)
    
    
class ComputationStrategy(ABC):
    def __init__(self, hamiltonian: LatticeHamiltonian, spin=1/2):
        self.hamiltonian = hamiltonian
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
    
class DiagonalizationEngine(ComputationStrategy):
    def build_basis(self, a=1):
        self.basis = spin_basis_1d(L=self.hamiltonian.L, a=a)
        
    def build_hamiltonian(self, t):
        static = [[self.hamiltonian(t)[i][4], self.hamiltonian(t)[i]] for i
                  in range(self.hamiltonian.L)]
        print(static)
    
    def run_calculation(self, t: float = 0.0):
        # Construct Hamiltonian
        H = self.hamiltonian.construct_hamiltonian(t)
        
        # Perform exact diagonalization (placeholder)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        return eigenvalues, eigenvectors
    
class DMRGEngine(ComputationStrategy):
    def run_calculation(self, t: float = 0.0):
        # Construct Hamiltonian
        H = self.hamiltonian_builder.construct_hamiltonian(t)
        
        # Perform DMRG (placeholder)
        # This would use Block2 or other DMRG library
        print("DMRG calculation placeholder")
        
        return H  # Simplified return


if __name__ == "__main__":
    # Example usage
    hamiltonian = LatticeHamiltonian(L=4)
    hamiltonian.terms = [['xx', 1, 'NN'], ['yy', 1, 'NN'], ['z', 2, np.infty]]
    hamiltonian = hamiltonian.construct_graph()
    
    print(hamiltonian(1))
    
    computation = DiagonalizationEngine(hamiltonian)
    computation.build_basis()
    computation.build_hamiltonian(0.0)
    #computation.run_calculation(0.0)