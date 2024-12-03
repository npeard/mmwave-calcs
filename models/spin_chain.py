import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Any, Dict
import quspin
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.tools.Floquet import Floquet, Floquet_t_vec  # Floquet Hamiltonian
from quspin.operators import hamiltonian  # Hamiltonians
from quspin.tools.misc import matvec
from scipy.linalg import expm


class LatticeHamiltonian:
    def __init__(self, L: int = None, interaction_dict: dict = None):
        self.L = L  # number of sites
        self.interaction_dict: Dict[str, List[List[Any]]] = interaction_dict or {}
    
    def __call__(self, t):
        return {
            op: [[f(t) if callable(f) else f for f in interaction]
                 for interaction in interactions]
            for op, interactions in self.interaction_dict.items()
        }
    
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
                        graph = [[lambda t, i=i, j=i + 1: strength(t, i, j), i,
                                  i + 1]
                                 for i in range(L - 1)]
                    else:
                        graph = [[strength, i, i + 1] for i in
                                 range(L - 1)]
                
                # Next-Nearest Neighbor (NNN) interactions
                elif alpha == 'nnn':
                    if callable(strength):
                        graph = [[lambda t, i=i, j=i + 2: strength(t, i, j), i,
                                  i + 2]
                                 for i in range(L - 2)]
                    else:
                        graph = [[strength, i, i + 2] for i in
                                 range(L - 2)]
                
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
                    graph = [[lambda t, i=i: strength(t, i), i]
                             for i in range(L)]
                else:
                    # Constant on-site term
                    graph = [[strength, i] for i in range(L)]
                
                # Add to the dictionary for this operator
                new_interaction_dict[operator].extend(graph)
            
            else:
                if len(operator) != 2:
                    raise ValueError(f"Two-site operation requires two-site "
                                     f"operator: {operator}")
                
                # General long-range interaction
                if callable(strength):
                    # Time and site-dependent interaction
                    graph = [
                        [lambda t, i=i, j=j: strength(t, i, j), i, j]
                        for i in range(L)
                        for j in range(L)
                    ]
                else:
                    # Constant interaction strength
                    graph = [[strength, i, j] for i in range(L)
                             for j in range(L)]
                
                # Add to the dictionary for this operator
                new_interaction_dict[operator].extend(graph)
        
        # Create a new Hamiltonian with the constructed interaction dictionary
        return cls(L, new_interaction_dict)
    
    
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
        # Put our Hamiltonian into QuSpin format
        static = [[key, self.hamiltonian(t)[key]] for key in self.hamiltonian(t).keys()]
        # Create QuSpin Hamiltonian
        H = quspin.operators.hamiltonian(static, [], basis=self.basis)
        
        return H
    
    def run_calculation(self, t: float = 0.0):
        pass
    
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
    terms = [['xx', 1, 'nn'], ['yy', 1, 'nn'], ['z', 2, np.inf],
             ['xx', 3, 'nn']]
    hamiltonian = LatticeHamiltonian.from_interactions(4, terms)
    
    print(hamiltonian(1))
    
    computation = DiagonalizationEngine(hamiltonian)
    computation.build_basis()
    computation.build_hamiltonian(0.0)
    #computation.run_calculation(0.0)