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
        Dictionary of parameter names and initial values
    func : callable
        Function that takes (t, *args, **params) where:
        - t is the time
        - *args are site indices (i, j, etc.)
        - **params are the parameters to optimize
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
            Time at which to evaluate
        *args : tuple
            Additional arguments (site indices)
        """
        # Collect parameters into a dict
        params = {name: param for name, param in self.named_parameters()}
        # Call function with time first, then args, then params
        return self.func(t, *args, **params)

class TorchSpinOperator:
    """
    Class to handle spin operators with PyTorch sparse tensors.
    """
    def __init__(self, num_sites: int, spin: str = '1/2'):
        """
        Initialize spin operator handler with sparse matrices.

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
        self.dim = int(2*float(eval(self.spin)) + 1)

    def _get_spin_operators(self) -> Dict[str, torch.Tensor]:
        """Get the spin operators as PyTorch sparse tensors."""
        if self.spin != '1/2':
            raise ValueError("Currently only spin-1/2 is supported")

        # Create sparse Pauli matrices
        # σx
        x_indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x_values = torch.ones(2, dtype=torch.complex128)
        sigma_x = torch.sparse_coo_tensor(x_indices.t(), x_values, (2, 2))

        # σy
        y_indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        y_values = torch.tensor([-1j, 1j], dtype=torch.complex128)
        sigma_y = torch.sparse_coo_tensor(y_indices.t(), y_values, (2, 2))

        # σz
        z_indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
        z_values = torch.tensor([1, -1], dtype=torch.complex128)
        sigma_z = torch.sparse_coo_tensor(z_indices.t(), z_values, (2, 2))

        # σ+ and σ-
        plus_indices = torch.tensor([[0], [1]], dtype=torch.long)
        plus_values = torch.ones(1, dtype=torch.complex128)
        sigma_plus = torch.sparse_coo_tensor(plus_indices, plus_values, (2, 2))

        minus_indices = torch.tensor([[1], [0]], dtype=torch.long)
        minus_values = torch.ones(1, dtype=torch.complex128)
        sigma_minus = torch.sparse_coo_tensor(minus_indices, minus_values, (2, 2))

        return {
            'X': sigma_x,
            'Y': sigma_y,
            'Z': sigma_z,
            '+': sigma_plus,
            '-': sigma_minus
        }

    def _sparse_kron(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute the Kronecker product of two sparse tensors.

        Parameters
        ----------
        A : torch.Tensor
            First sparse tensor
        B : torch.Tensor
            Second sparse tensor

        Returns
        -------
        torch.Tensor
            Kronecker product of A and B
        """
        if not A.is_sparse or not B.is_sparse:
            raise ValueError("Both tensors must be sparse")

        A_indices = A._indices()
        B_indices = B._indices()
        A_values = A._values()
        B_values = B._values()

        # Get dimensions
        A_rows, A_cols = A.shape
        B_rows, B_cols = B.shape

        # For each non-zero element (i,j) in A and (k,l) in B,
        # we get (i*nrows_B + k, j*ncols_B + l) in the output
        A_idx_list = [(i.item(), j.item()) for i, j in zip(A_indices[0], A_indices[1])]
        B_idx_list = [(i.item(), j.item()) for i, j in zip(B_indices[0], B_indices[1])]

        out_indices = []
        out_values = []

        for (i, j), a_val in zip(A_idx_list, A_values):
            for (k, l), b_val in zip(B_idx_list, B_values):
                out_indices.append([
                    i * B_rows + k,
                    j * B_cols + l
                ])
                out_values.append(a_val * b_val)

        if not out_indices:  # Handle empty matrices
            return torch.sparse_coo_tensor(
                size=(A_rows * B_rows, A_cols * B_cols),
                dtype=A.dtype
            )

        out_indices = torch.tensor(out_indices).t()
        out_values = torch.tensor(out_values, dtype=A.dtype)

        return torch.sparse_coo_tensor(
            out_indices, out_values,
            size=(A_rows * B_rows, A_cols * B_cols)
        )

    def get_site_operator(self, op_string: str, site: int) -> torch.Tensor:
        """
        Get sparse operator for a specific site using proper tensor products.
        Follows standard quantum mechanics convention where site 0 is the leftmost tensor factor.

        Parameters
        ----------
        op_string : str
            Operator type (X, Y, Z, +, -)
        site : int
            Site index

        Returns
        -------
        torch.Tensor
            Sparse tensor representing the operator on the full Hilbert space
        """
        if site >= self.num_sites:
            raise ValueError(f"Site index {site} exceeds system size {self.num_sites}")

        op = self.operators.get(op_string.upper())
        if op is None:
            raise ValueError(f"Unknown operator: {op_string}")

        # Create identity matrix for single site
        id_indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        id_values = torch.ones(2, dtype=torch.complex128)
        identity = torch.sparse_coo_tensor(id_indices, id_values, (2, 2))

        # Build operator using tensor products
        # For a system with N sites, we want: I⊗I⊗...⊗O⊗...⊗I
        # where O is at position 'site' counting from the left

        # Start with leftmost operator (site 0)
        result = identity if site > 0 else op

        # Add remaining operators from left to right
        for i in range(1, self.num_sites):
            next_op = op if i == site else identity
            result = self._sparse_kron(result, next_op)

        return result

    def get_interaction_operator(self, op_string: str, site1: int, site2: int) -> torch.Tensor:
        """
        Get sparse operator for interaction between two sites using tensor products.
        Follows standard quantum mechanics convention where site 0 is the leftmost tensor factor.

        Parameters
        ----------
        op_string : str
            Two-character string specifying operators (e.g., 'XX', 'YZ')
        site1, site2 : int
            Site indices

        Returns
        -------
        torch.Tensor
            Sparse tensor representing the interaction operator
        """
        if len(op_string) != 2:
            raise ValueError(f"Expected two-site operator, got {op_string}")
        if site1 >= self.num_sites or site2 >= self.num_sites:
            raise ValueError(f"Site indices {site1}, {site2} exceed system size {self.num_sites}")
        if site1 == site2:
            raise ValueError(f"Sites must be different, got {site1}, {site2}")

        # Get individual operators
        op1 = self.operators.get(op_string[0].upper())
        op2 = self.operators.get(op_string[1].upper())
        if op1 is None or op2 is None:
            raise ValueError(f"Unknown operator(s) in {op_string}")

        # Create identity matrix for single site
        id_indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        id_values = torch.ones(2, dtype=torch.complex128)
        identity = torch.sparse_coo_tensor(id_indices, id_values, (2, 2))

        # Build the full operator site by site
        result = identity if 0 not in (site1, site2) else (op1 if site1 == 0 else op2)

        for i in range(1, self.num_sites):
            next_op = identity
            if i == site1:
                next_op = op1
            elif i == site2:
                next_op = op2
            result = self._sparse_kron(result, next_op)

        return result

    def get_operator_sparse(self, op_string: str, sites: List[int]) -> torch.Tensor:
        """
        Get sparse operator for a specific operator string and site indices.

        Parameters
        ----------
        op_string : str
            Operator type (X, Y, Z, +, -)
        sites : List[int]
            Site indices
        """
        if len(sites) == 1:
            return self.get_site_operator(op_string, sites[0])
        elif len(sites) == 2:
            return self.get_interaction_operator(op_string, *sites)
        else:
            raise ValueError(f"Invalid number of sites: {len(sites)}")

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
