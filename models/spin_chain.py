import numpy as np
from abc import ABC, abstractmethod

class LatticeHamiltonian:
    def __init__(self, graph_data, L):
        self.L = L # number of sites
        J=1
        D=1
        alpha1=0
        alpha2=np.infty
        self.terms = [[J, alpha1], [D, alpha2]]
        self.graph_data = graph_data
    
    def __call__(self, t):
        return [[f(t) if callable(f) else f for f in row] for row in
                self.graph_data]
    
    def construct_graph(self, t=None):
        '''build the graph of interactions between sites'''
        for term in self.terms:
            alpha = term[1]
            if isinstance(alpha, str):
                # long-range with cutoff, like NN or NNN
                graph = get_graph_string(alpha)
            elif alpha == np.infty:
                # on-site interaction
                U = term[0]
                if callable(U):
                    graph = [[lambda t, i=i: U(t, i), i] for i in range(self.L)]
                else:
                    graph = [[U, i] for i in range(self.L)]
            else:
                # long-range interaction
                J = term[0]
                if callable(J):
                    graph = [[lambda t, i=i, j=j: J(t, i, j), i, j] for i in
                             range(self.L) for j in range(self.L)]
                else:
                    graph = [[J, i, j] for i in range(self.L) for j in
                             range(self.L)]
                self.graph_data.append(graph)
        return LatticeHamiltonian(self.graph_data, self.L)
    
    
class ComputationStrategy(ABC):
    def __init__(self, hamiltonian: LatticeHamiltonian):
        self.hamiltonian = hamiltonian
    
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
