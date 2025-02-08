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
                 spin: str = '1/2',
                 external_parameters: Optional[List[TorchParameter]] = None):
        """
        Initialize optimizer.

        Parameters
        ----------
        graph : LatticeGraph or TorchLatticeGraph
            Graph to optimize
        target_graph : LatticeGraph or TorchLatticeGraph, optional
            Target graph to optimize towards
        spin : str, optional
            Spin representation to use
        external_parameters : List[TorchParameter], optional
            Additional TorchParameter instances to optimize (e.g., time parameters)
        """
        self.graph = graph
        self.spin = spin
        self.num_sites = graph.num_sites
        self.external_parameters = external_parameters or []

        if target_graph is not None:
            self.target_hamiltonian = self.build_hamiltonian(target_graph, t=0)
        else:
            self.target_hamiltonian = None

        # Collect unique TorchParameter instances from both graph and external parameters
        self.torch_parameters = self._collect_unique_torch_parameters()
        self.optimizer = None

    def _collect_unique_torch_parameters(self) -> List[TorchParameter]:
        """Collect unique TorchParameter instances from the graph and external parameters."""
        seen = set()
        unique_params = []

        # Collect from graph
        for interactions in self.graph.interaction_dict.values():
            for interaction in interactions:
                strength = interaction[0]
                if isinstance(strength, TorchParameter):
                    if id(strength) not in seen:
                        seen.add(id(strength))
                        unique_params.append(strength)

        # Collect from external parameters
        for param in self.external_parameters:
            if isinstance(param, TorchParameter):
                if id(param) not in seen:
                    seen.add(id(param))
                    unique_params.append(param)

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
        state_multiplicity = int(2*float(eval(self.spin)) + 1)
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

    def compute_fidelity_loss(self, U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
        """Compute fidelity loss between two unitary operators."""
        overlap = torch.abs(torch.trace(U1.conj().T @ U2))
        norm = torch.sqrt(torch.abs(torch.trace(U1.conj().T @ U1)) *
                         torch.abs(torch.trace(U2.conj().T @ U2)))
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

    def build_hamiltonian_list(self, params: List[float]) -> List[torch.Tensor]:
        """
        Build a list of Hamiltonians evaluated at different parameter values.

        Parameters
        ----------
        params : List[float]
            List of parameter values (could be times or other parameters)
            at which to evaluate the Hamiltonians

        Returns
        -------
        List[torch.Tensor]
            List of Hamiltonian matrices
        """
        return [self.build_hamiltonian(self.graph, param) for param in params]

    def compute_floquet_unitary(self, dt_list: List[float], hamiltonians: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the Floquet unitary evolution operator by applying each Hamiltonian
        for its corresponding time step.

        Parameters
        ----------
        dt_list : List[float]
            List of time steps
        hamiltonians : List[torch.Tensor]
            List of Hamiltonian matrices to apply

        Returns
        -------
        torch.Tensor
            The Floquet unitary evolution operator
        """
        if len(dt_list) != len(hamiltonians):
            raise ValueError("dt_list and hamiltonians must have the same length")

        # Get dimension from first Hamiltonian
        dim = hamiltonians[0].shape[0]
        U = torch.zeros((dim, dim), dtype=torch.complex128)

        # For each initial state
        for i in range(dim):
            # Initialize state vector with 1 at position i
            psi = torch.zeros(dim, dtype=torch.complex128)
            psi[i] = 1.0

            # Apply each Hamiltonian evolution
            for dt, H in zip(dt_list, hamiltonians):
                # Compute evolution operator exp(-i*H*dt)
                evolution = torch.matrix_exp(-1j * H * dt)
                # Apply to state vector
                psi = evolution @ psi

            # Store result as column i of U
            U[:, i] = psi

        return U

    def get_floquet_evolution(self, dt_list: List[float], params: List[float]) -> torch.Tensor:
        """
        Compute Floquet evolution operator for a sequence of Hamiltonians and time steps.
        Interface similar to get_quspin_floquet_hamiltonian in DiagonEngine.

        Parameters
        ----------
        dt_list : List[float]
            List of time steps
        params : List[float]
            List of parameter values at which to evaluate Hamiltonians

        Returns
        -------
        torch.Tensor
            The Floquet unitary evolution operator
        """
        hamiltonians = self.build_hamiltonian_list(params)
        # To allow for delta pulses (pure phase rotations), turn all dt=0 elements
        # to dt=1 for numerical time evolution with a supplied phase value
        dt_list = [dt if dt != 0 else 1 for dt in dt_list]
        return self.compute_floquet_unitary(dt_list, hamiltonians)

    def optimize_floquet_sequence(self,
                                dt_params: List[Union[float, TorchParameter, Callable]],
                                param_list: List[str],
                                target_U: Optional[torch.Tensor] = None,
                                num_epochs: int = 100,
                                verbose: bool = True) -> List[float]:
        """
        Optimize the Floquet sequence parameters to match a target unitary.

        Parameters
        ----------
        dt_params : List[Union[float, TorchParameter, Callable]]
            List of time step parameters. Each element can be:
            - A float (fixed time)
            - A TorchParameter (optimizable time)
            - A callable that returns a tensor (computed time)
        param_list : List[str]
            List of parameter labels for each step
        target_U : torch.Tensor, optional
            Target unitary operator. If None, uses the target graph to compute it
        num_epochs : int, optional
            Number of optimization steps
        verbose : bool, optional
            Whether to print progress

        Returns
        -------
        List[float]
            List of loss values during training
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set up. Call setup_optimizer first.")

        losses = []
        target_U_computed = None

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            # Evaluate time parameters for this step
            dt_list = []
            for dt in dt_params:
                if isinstance(dt, TorchParameter):
                    dt_list.append(dt(0))
                elif callable(dt):
                    dt_list.append(dt())
                else:
                    dt_list.append(dt)

            floquet_period = sum(dt_list)

            # If no target_U provided, compute it from target graph
            if target_U is None:
                if self.target_hamiltonian is not None:
                    # Only compute target_U once and detach it from computation graph
                    if target_U_computed is None:
                        target_U_computed = torch.matrix_exp(-1j * self.target_hamiltonian * floquet_period).detach()
                    target_U = target_U_computed
                else:
                    raise ValueError("Either target_U or target_graph must be provided")

            # Get current unitary
            U = self.get_floquet_evolution(dt_list, param_list)

            # Compute loss
            loss = self.compute_fidelity_loss(U, target_U)
            losses.append(loss.item())

            # Backpropagate and optimize
            loss.backward()#retain_graph=True)
            self.optimizer.step()

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                curr_times = [dt.item() if torch.is_tensor(dt) else dt for dt in dt_list]
                print(f"Current times: {curr_times}")

        return losses

if __name__ == "__main__":
    # Example usage of time-dependent functions
    def DM_z_period4_torch(t, i, phase_factor=torch.tensor(np.pi/2, dtype=torch.float64)):
        """DM interaction with period 4 pattern."""
        phase = phase_factor * (i % 4)
        if t == "+DM":
            return phase
        elif t == "-DM":
            return -phase
        else:
            return torch.tensor(0.0, dtype=torch.float64)

    def XY_z_period4_torch(t, i, base_phase=torch.tensor(np.pi, dtype=torch.float64),
                        phase_factor=torch.tensor(3*np.pi/2, dtype=torch.float64)):
        """XY interaction with period 4 pattern."""
        phase = base_phase - phase_factor * (i % 4)
        if t == "+XY":
            return phase
        elif t == "-XY":
            return -phase
        else:
            return torch.tensor(0.0, dtype=torch.float64)

    def native_torch(t, i, j, coupling=torch.tensor(0.5, dtype=torch.float64)):
        """Native interaction strength."""
        if t in ["+DM", "-DM", "+XY", "-XY"]:
            return torch.tensor(0.0, dtype=torch.float64)
        else:
            return coupling

    # Function for time parameters that just returns the value
    def time_param(t, value=torch.tensor(0.1, dtype=torch.float64)):
        """Simple function to return an optimizable time parameter."""
        return value

    # Create TorchParameter instances for interactions
    dm_params = {'phase_factor': np.pi/2}
    dm_interaction = TorchParameter(dm_params, DM_z_period4_torch)

    xy_params = {
        'base_phase': np.pi,
        'phase_factor': 3*np.pi/2
    }
    xy_interaction = TorchParameter(xy_params, XY_z_period4_torch)

    native_params = {'coupling': 0.5}
    native_interaction = TorchParameter(native_params, native_torch)

    # Create TorchParameter instances for time parameters
    tJ = TorchParameter({'value': 0.1}, time_param)
    tD = TorchParameter({'value': 0.2}, time_param)
    tmJ = TorchParameter({'value': 0.15}, time_param)

    # Collect time parameters
    time_parameters = [tJ, tD, tmJ]

    # set number of sites
    num_sites = 4

    # Example terms using the TorchParameter instances
    terms = [
        ['XX', native_interaction, 'nn'],
        ['yy', native_interaction, 'nn'],
        ['z', dm_interaction, np.inf],
        ['z', xy_interaction, np.inf]
    ]
    graph = TorchLatticeGraph.from_torch_interactions(num_sites, terms, pbc=True)

    # Target graph
    DM_terms = [['xy', 1/2, 'nn'], ['yx', -1/2, 'nn']]
    DM_graph = TorchLatticeGraph.from_torch_interactions(num_sites, DM_terms, pbc=True)

    # Create optimizer with time parameters
    optimizer = FloquetOptimizer(graph, target_graph=DM_graph, external_parameters=time_parameters)
    print("\nCollected torch parameters:", optimizer.torch_parameters)
    print("Parameters to optimize:", [p for param in optimizer.torch_parameters for p in param.parameters()])

    optimizer.setup_optimizer(optimizer_class=torch.optim.Adam, lr=0.01)

    # Define sequence and optimize
    param_list = ["nat", "+DM", "nat", "+XY", "nat", "-XY", "nat", "-DM", "nat"]
    dt_params = [tJ, 0, tD, 0, lambda: 2 * tmJ(0), 0, tD, 0, tJ]  # Now passing parameters instead of evaluated values

    print("\nStarting optimization...")
    print("Initial parameters:")
    print(f"tJ: {tJ(0).item():.6f}")
    print(f"tD: {tD(0).item():.6f}")
    print(f"tmJ: {tmJ(0).item():.6f}")

    losses = optimizer.optimize_floquet_sequence(dt_params, param_list, num_epochs=10)

    print("\nFinal parameters:")
    print(f"tJ: {tJ(0).item():.6f}")
    print(f"tD: {tD(0).item():.6f}")
    print(f"tmJ: {tmJ(0).item():.6f}")

    print("\nFinal loss:", losses[-1])
