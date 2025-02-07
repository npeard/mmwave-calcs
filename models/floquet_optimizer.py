"""
PyTorch-compatible optimization for Floquet systems.

This module provides PyTorch-based implementations for optimizing
time-dependent parameters in Floquet systems while maintaining
compatibility with the existing LatticeGraph infrastructure.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Union, Optional, Callable
from models.spin_chain import LatticeGraph

class TorchParameter(nn.Module):
    """
    Wrapper for time-dependent parameters that need to be optimized using PyTorch.

    Parameters
    ----------
    params : dict
        Dictionary of parameter names and their initial values
    func : callable
        Function that takes the parameters dictionary and a time value as input
        and returns the value of the parameter at that time
    """
    def __init__(self, params: dict, func: callable):
        super().__init__()

        # Register each parameter with PyTorch
        for name, value in params.items():
            setattr(self, name, nn.Parameter(torch.tensor(value, dtype=torch.float64)))

        self.func = func

    def forward(self, t: float, *args) -> torch.Tensor:
        """
        Compute the value of the parameter at time t.

        Parameters
        ----------
        t : float
            Time at which to evaluate the parameter
        *args : tuple
            Additional arguments passed to the function (ignored for compatibility)
        """
        params = {name: param for name, param in self.named_parameters()}
        return self.func(params, t)

class TorchSpinOperator:
    """
    Class to handle spin operators with PyTorch tensors.
    """
    def __init__(self, num_sites: int, spin: str = '1/2'):
        """
        Initialize spin operator handler.

        Parameters
        ----------
        num_sites : int
            Number of sites in the system
        spin : str, optional
            Spin representation ('1/2' only for now)
        """
        self.num_sites = num_sites
        self.spin = spin
        self.operators = self._get_spin_operators()
        self.dim = 2  # for spin-1/2

    def _get_spin_operators(self) -> Dict[str, torch.Tensor]:
        """Get the spin operators as PyTorch tensors."""
        if self.spin != '1/2':
            raise ValueError("Currently only spin-1/2 is supported")

        # Define Pauli matrices
        s_plus = torch.tensor([[0., 1.], [0., 0.]], dtype=torch.complex128)
        s_minus = torch.tensor([[0., 0.], [1., 0.]], dtype=torch.complex128)
        sz = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex128) / 2

        # Construct X and Y from raising/lowering operators
        sx = (s_plus + s_minus) / 2
        sy = (s_plus - s_minus) / (2j)

        return {
            'X': sx, 'Y': sy, 'Z': sz,
            '+': s_plus, '-': s_minus,
            'x': sx, 'y': sy, 'z': sz  # Allow lowercase
        }

    def get_site_operator(self, op_string: str, site: int) -> torch.Tensor:
        """
        Get operator for a specific site.

        Parameters
        ----------
        op_string : str
            Operator type (X, Y, Z, +, -)
        site : int
            Site index
        """
        op = self.operators[op_string]

        # Build up the operator using kronecker products
        result = torch.eye(1, dtype=torch.complex128)

        # Identity for sites before target
        for _ in range(site):
            result = torch.kron(result, torch.eye(self.dim, dtype=torch.complex128))

        # Operator at target site
        result = torch.kron(result, op)

        # Identity for sites after target
        for _ in range(site + 1, self.num_sites):
            result = torch.kron(result, torch.eye(self.dim, dtype=torch.complex128))

        return result

    def get_interaction_operator(self, op_string: str, site1: int, site2: int) -> torch.Tensor:
        """
        Get operator for interaction between two sites.

        Parameters
        ----------
        op_string : str
            Two-character string specifying operators (e.g., 'XX', 'YZ')
        site1, site2 : int
            Site indices
        """
        if len(op_string) != 2:
            raise ValueError(f"Expected two-site operator, got {op_string}")

        op1 = self.get_site_operator(op_string[0], site1)
        op2 = self.get_site_operator(op_string[1], site2)

        return op1 @ op2

class TorchLatticeGraph(LatticeGraph):
    """
    Extension of LatticeGraph that properly handles PyTorch parameters.
    """
    @classmethod
    def from_torch_interactions(cls, num_sites: int, terms: list[list[Any]],
                              pbc: bool = False):
        """
        Create a LatticeGraph with proper handling of PyTorch parameters.

        Parameters
        ----------
        num_sites : int
            Number of sites in the lattice
        terms : list[list[Any]]
            List of [operator, strength, range] terms where strength can be:
            - A number
            - A TorchParameter instance
            - A function taking (t, i) for single-site or (t, i, j) for two-site terms
        pbc : bool, optional
            Whether to use periodic boundary conditions
        """
        graph = cls(num_sites)

        for term in terms:
            operator, strength, alpha = term
            print(f"\nProcessing term: {term}")
            print(f"- Strength type: {type(strength)}")
            if isinstance(strength, TorchParameter):
                print(f"- TorchParameter params: {list(strength.named_parameters())}")

            if not isinstance(operator, str):
                raise ValueError(f"Invalid operator type: {type(operator)}")

            if operator not in graph.interaction_dict:
                graph.interaction_dict[operator] = []

            # Handle different types of interactions
            if isinstance(alpha, str):
                if alpha == 'nn':  # Nearest neighbor
                    for i in range(num_sites - 1 + int(pbc)):
                        j = (i + 1) % num_sites
                        graph.interaction_dict[operator].append([strength, i, j])
                elif alpha == 'nnn':  # Next-nearest neighbor
                    for i in range(num_sites - 2 + 2*int(pbc)):
                        j = (i + 2) % num_sites
                        graph.interaction_dict[operator].append([strength, i, j])
                else:
                    raise ValueError(f"Invalid range string: {alpha}")
            elif isinstance(alpha, (int, float)):
                if alpha == np.inf:  # On-site terms
                    for i in range(num_sites):
                        graph.interaction_dict[operator].append([strength, i])
                elif alpha == 0:  # Nearest-neighbor
                    for i in range(num_sites - 1 + int(pbc)):
                        j = (i + 1) % num_sites
                        graph.interaction_dict[operator].append([strength, i, j])
                else:  # Power-law decay
                    for i in range(num_sites):
                        for j in range(i + 1, num_sites):
                            if not pbc and (j >= num_sites):
                                continue
                            j = j % num_sites
                            dist = min(abs(i - j), num_sites - abs(i - j)) if pbc else abs(i - j)
                            decay = 1.0 / (dist ** alpha) if dist > 0 else 0.0

                            if isinstance(strength, TorchParameter):
                                # Keep the TorchParameter instance as is, just scale its output
                                scaled_strength = TorchParameter(
                                    {name: param.item() * decay for name, param in strength.named_parameters()},
                                    strength.func
                                )
                                graph.interaction_dict[operator].append([scaled_strength, i, j])
                            elif callable(strength):
                                # Only wrap non-TorchParameter callables in lambda
                                graph.interaction_dict[operator].append([
                                    lambda t, s=strength, d=decay: s(t) * d,
                                    i, j
                                ])
                            else:
                                # Simple scalar multiplication
                                graph.interaction_dict[operator].append([strength * decay, i, j])

        return graph

    def evaluate_interaction(self, interaction: list, t: float) -> Union[float, torch.Tensor]:
        """
        Evaluate an interaction term at time t.

        Parameters
        ----------
        interaction : list
            Interaction term from interaction_dict
        t : float
            Time at which to evaluate
        """
        strength = interaction[0]
        sites = interaction[1:]

        if isinstance(strength, TorchParameter):
            return strength(t, *sites)
        elif callable(strength):
            return strength(t, *sites)
        else:
            return strength

class FloquetOptimizer:
    """
    Optimizer for Floquet systems using PyTorch automatic differentiation.
    """
    def __init__(self, graph: Union[LatticeGraph, TorchLatticeGraph],
                 target_graph: Optional[Union[LatticeGraph, TorchLatticeGraph]] = None,
                 spin: str = '1/2'):
        """
        Initialize the optimizer.

        Parameters
        ----------
        graph : LatticeGraph or TorchLatticeGraph
            The graph containing the interactions to optimize
        target_graph : LatticeGraph or TorchLatticeGraph, optional
            Target graph to optimize towards
        spin : str, optional
            Spin representation ('1/2' only for now)
        """
        self.graph = graph
        self.spin = spin
        self.num_sites = graph.num_sites

        if target_graph is not None:
            self.target_hamiltonian = self.build_hamiltonian(target_graph, t=0)
        else:
            self.target_hamiltonian = None

        # Collect unique TorchParameter instances
        self.torch_parameters = self._collect_unique_torch_parameters()
        self.optimizer = None

    def _collect_unique_torch_parameters(self) -> List[TorchParameter]:
        """Collect unique TorchParameter instances from the graph."""
        seen = set()
        unique_params = []
        for interactions in self.graph.interaction_dict.values():
            for interaction in interactions:
                strength = interaction[0]
                if isinstance(strength, TorchParameter):
                    # Use id to identify unique instances
                    if id(strength) not in seen:
                        seen.add(id(strength))
                        unique_params.append(strength)
        return unique_params

    def build_hamiltonian(self, graph: Union[LatticeGraph, TorchLatticeGraph],
                         t: float) -> torch.Tensor:
        """
        Build Hamiltonian using PyTorch operations.

        Parameters
        ----------
        graph : LatticeGraph or TorchLatticeGraph
            Graph containing the interactions
        t : float
            Time at which to evaluate
        """

        self.spin_operator = TorchSpinOperator(self.num_sites, self.spin)
        state_multiplicity = 2*eval(self.spin)+1
        H = torch.zeros((state_multiplicity**self.num_sites, state_multiplicity**self.num_sites), dtype=torch.complex128)

        for op_type, interactions in graph.interaction_dict.items():
            for interaction in interactions:
                if isinstance(graph, TorchLatticeGraph):
                    strength = graph.evaluate_interaction(interaction, t)
                else:
                    strength = interaction[0]
                    if callable(strength):
                        strength = strength(t)

                sites = interaction[1:]

                # Get the operator tensor
                if len(sites) == 1:
                    op = self.spin_operator.get_site_operator(op_type, sites[0])
                else:
                    op = self.spin_operator.get_interaction_operator(op_type, *sites)

                H = H + strength * op

        return H

    def setup_optimizer(self, optimizer_class=torch.optim.Adam, **optimizer_kwargs):
        """Set up the PyTorch optimizer."""
        parameters = []
        for torch_param in self.torch_parameters:
            parameters.extend(torch_param.parameters())

        self.optimizer = optimizer_class(parameters, **optimizer_kwargs)

    def compute_fidelity_loss(self, H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
        """Compute fidelity loss between two Hamiltonians."""
        overlap = torch.abs(torch.trace(H1.conj().T @ H2))
        norm = torch.sqrt(torch.abs(torch.trace(H1.conj().T @ H1)) *
                         torch.abs(torch.trace(H2.conj().T @ H2)))
        return 1 - overlap / norm

    def optimize_step(self, t: float,
                     loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> float:
        """
        Perform one optimization step.

        Parameters
        ----------
        t : float
            Time at which to evaluate
        loss_fn : callable, optional
            Custom loss function. If None, uses fidelity loss
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set up. Call setup_optimizer first.")

        self.optimizer.zero_grad()

        # Get current Hamiltonian
        H = self.build_hamiltonian(self.graph, t)

        # Compute loss
        if loss_fn is None:
            if self.target_hamiltonian is None:
                raise ValueError("No target Hamiltonian set and no custom loss function provided")
            loss = self.compute_fidelity_loss(H, self.target_hamiltonian)
        else:
            loss = loss_fn(H)

        # Backpropagate and optimize
        loss.backward()
        self.optimizer.step()

        return loss.item()

def fourier_component(params: dict, t: float) -> torch.Tensor:
    """Example time-dependent function that computes a Fourier component."""
    return params['amplitude'] * torch.sin(params['frequency'] * t)

if __name__ == "__main__":
    # Create a parameterized time-dependent function
    initial_params = {
        'amplitude': 1.0,
        'frequency': 2.0
    }
    torch_callable = TorchParameter(initial_params, fourier_component)
    print("\nCreated TorchParameter:")
    print("- Parameters:", list(torch_callable.named_parameters()))
    print("- Is callable:", callable(torch_callable))

    # Create target graph (Dzyaloshinskii-Moriya interaction)
    DM_terms = [['xy', 1/2, 'nn'], ['yx', -1/2, 'nn']]
    DM_graph = TorchLatticeGraph.from_torch_interactions(4, DM_terms, pbc=True)

    # Create graph with the torch parameter
    terms = [['XX', torch_callable, 0], ['YY', 1.0, 0]]
    graph = TorchLatticeGraph.from_torch_interactions(4, terms)

    print("\nGraph interaction dict:")
    for op_type, interactions in graph.interaction_dict.items():
        print(f"\nOperator {op_type}:")
        for interaction in interactions:
            print(f"- Interaction: {interaction}")
            if callable(interaction[0]):
                print(f"  - Strength type: {type(interaction[0])}")
                if isinstance(interaction[0], TorchParameter):
                    print(f"  - Parameters: {list(interaction[0].named_parameters())}")

    # Create optimizer
    optimizer = FloquetOptimizer(graph, target_graph=DM_graph)
    print("\nCollected torch parameters:", optimizer.torch_parameters)
    print("Parameters to optimize:", [p for param in optimizer.torch_parameters for p in param.parameters()])

    optimizer.setup_optimizer(optimizer_class=torch.optim.Adam, lr=0.01)

    # Optimization loop
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = optimizer.optimize_step(t=0.0)
        print(f"Epoch {epoch}, Loss: {loss}")
