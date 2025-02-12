import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from models.torch_chain import TorchLatticeGraph, TorchParameter, TorchSpinOperator

def sparse_matrix_exp(matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the matrix exponential of a sparse matrix using the scaling and squaring method.
        Maintains sparsity throughout computation.

        Parameters
        ----------
        matrix : torch.Tensor
            Sparse matrix to exponentiate

        Returns
        -------
        torch.Tensor
            Matrix exponential of the input matrix as a sparse tensor
        """
        if not matrix.is_sparse:
            raise ValueError("Input matrix must be sparse")

        # Scaling and squaring method
        max_power = 10
        scale = 2 ** max_power
        scaled_matrix = matrix / scale

        # Create sparse identity matrix
        dim = matrix.shape[0]
        indices = torch.arange(dim, dtype=torch.long)
        id_indices = torch.stack([indices, indices])
        id_values = torch.ones(dim, dtype=matrix.dtype)
        exp_matrix = torch.sparse_coo_tensor(id_indices, id_values, matrix.shape)

        # Compute matrix exponential using Taylor series
        # exp(A) = I + A + A^2/2! + A^3/3! + ...
        current_term = exp_matrix
        factorial = 1
        for k in range(1, max_power + 1):
            factorial *= k
            current_term = torch.sparse.mm(current_term, scaled_matrix) / factorial
            exp_matrix = exp_matrix + current_term

        # Square the result max_power times
        for _ in range(max_power):
            exp_matrix = torch.sparse.mm(exp_matrix, exp_matrix)

        return exp_matrix


class FloquetProgram(ABC):
    """
    Base class for defining a Floquet program.
    A Floquet program specifies the native Hamiltonian terms and target evolution.
    """
    def __init__(self, num_sites: int, spin='1/2', device: Optional[str] = None):
        """
        Initialize a Floquet program.

        Parameters
        ----------
        num_sites : int
            Number of sites in the system
        spin : str
            Spin representation (e.g. '1/2', '1', etc.)
        device : str, optional
            Device to run computations on ('cuda' or 'cpu'). If None, will use CUDA if available.
        """
        # Hamiltonian parameters
        self.num_sites = num_sites
        self.spin = spin
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize operator cache
        self._operator_cache = {}
        self._spin_operator = TorchSpinOperator(self.num_sites, self.spin)

        # Build graphs and cache operators
        self._native_graph = self._build_native_graph()
        self._target_graph = self._build_target_graph()
        self._cache_operators(self._native_graph)
        self._cache_operators(self._target_graph)

        # Pulse program parameters, lists of pulse parameters (can be times or strings)
        # and associated evolution time for that pulse
        self._build_sequence_timings()
        # Currently, I am only interested in optimizing Floquet sequences with a known structure,
        # so _build_sequence_timings() should initialize self._hamiltonian_sequence so that we only
        # call _build_hamiltonian once. If the sequence of applied Hamiltonians needs to be mutable,
        # then it will be necessary to make _build_hamiltonian more efficient.
        self._hamiltonian_sequence = None
        self.torch_params = self._collect_unique_torch_parameters()

    @abstractmethod
    def _build_native_graph(self):
        """
        Build the native Hamiltonian graph.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    @abstractmethod
    def _build_target_graph(self):
        """
        Build the target Hamiltonian graph.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    @abstractmethod
    def _build_sequence_timings(self):
        """
        Build the sequence timings.
        Must be implemented by subclasses.
        """
        # Currently, I am only interested in optimizing Floquet sequences with a known structure,
        # so _build_sequence_timings() should initialize self._hamiltonian_sequence so that we only
        # call _build_hamiltonian once. If the sequence of applied Hamiltonians needs to be mutable,
        # then it will be necessary to make _build_hamiltonian more efficient.
        raise NotImplementedError()

    def _cache_operators(self, graph: TorchLatticeGraph):
        """
        Cache all operators for a given graph.

        Parameters
        ----------
        graph : TorchLatticeGraph
            Graph to cache operators for
        """
        for op_string, interactions in graph.interaction_dict.items():
            for interaction in interactions:
                sites = interaction[1:]
                cache_key = (op_string, tuple(sites))
                if cache_key not in self._operator_cache:
                    self._operator_cache[cache_key] = self._spin_operator.get_operator_sparse(op_string, sites)

    def _build_hamiltonian(self, graph: TorchLatticeGraph, t: float) -> torch.Tensor:
        """
        Build Hamiltonian at time t using cached operators.
        Uses dense matrices throughout for autograd compatibility.

        Parameters
        ----------
        graph : TorchLatticeGraph
            Graph to build Hamiltonian from
        t : float
            Time at which to evaluate

        Returns
        -------
        torch.Tensor
            Dense Hamiltonian matrix
        """
        # Get dimension from number of sites
        dim = self._spin_operator.dim ** self.num_sites

        # Initialize dense Hamiltonian
        H = torch.zeros((dim, dim), dtype=torch.complex128, device=self.device)
        # Ideally, this would be a sparse matrix. But PyTorch autograd doesn't play
        # nice with sparse, complex-valued matrices. TODO: Find another solution?

        # Add all interaction terms using cached operators
        for op_string, interactions in graph.interaction_dict.items():
            for interaction in interactions:
                strength = graph.evaluate_interaction(interaction, t)
                sites = interaction[1:]

                # Get cached operator and convert to dense
                cache_key = (op_string, tuple(sites))
                op = self._operator_cache[cache_key].to_dense().to(self.device)

                # Scale operator by strength and add to H
                if isinstance(strength, torch.Tensor) or isinstance(strength, complex):
                    H = H + op * strength
                else:
                    H = H + op * float(strength)

        return H

    def _build_native_hamiltonian(self, t):
        """Build the native Hamiltonian including time-dependent control terms."""
        return self._build_hamiltonian(self._native_graph, t=t)

    def _build_target_hamiltonian(self):
        """Build the target Hamiltonian (time-independent)."""
        return self._build_hamiltonian(self._target_graph, t=0)

    def get_floquet_sequence(self):
        """
        Return the sequence of Hamiltonians and times for the Floquet system.
        Hamiltonians are kept sparse until needed.
        """
        hamiltonians = [self._build_native_hamiltonian(p) for p in self._pulse_parameters]
        delta_t = [dt if dt != 0 else 1 for dt in self._evolve_times]
        return delta_t, hamiltonians

    def get_target_unitary(self, floquet_period):
        """
        Compute the target unitary for the given Floquet period.
        """
        H_target = self._build_target_hamiltonian()
        U = torch.matrix_exp(-1j * H_target * floquet_period).detach()
        return U

    def _time_param(self, t, value=torch.tensor(0.1, dtype=torch.float64), name='t_*'):
        """Simple function to return an optimizable time parameter.
        Uses softplus to ensure positive values while maintaining smooth gradients."""
        return torch.nn.functional.softplus(value)

    def _collect_unique_torch_parameters(self) -> List[TorchParameter]:
        """Collect unique TorchParameter instances from the graph and external parameters."""
        seen = set()
        unique_params = []

        # Collect from graph
        for interactions in self._native_graph.interaction_dict.values():
            for interaction in interactions:
                strength = interaction[0]
                if isinstance(strength, TorchParameter):
                    if id(strength) not in seen:
                        seen.add(id(strength))
                        unique_params.append(strength)

        # Collect from external parameters
        for param in self._evolve_times:
            if isinstance(param, TorchParameter):
                if id(param) not in seen:
                    seen.add(id(param))
                    unique_params.append(param)

        return unique_params

    def _set_param_optimization(self, param_list, optimize=True):
        """Helper function to enable/disable optimization for parameters."""
        for param in param_list:
            for p in param.parameters():
                p.requires_grad = optimize

    def print_parameters(self, title: str):
        """
        Print all TorchParameters in an organized way.

        Parameters
        ----------
        title : str
            Title for the parameter printout
        """
        print(f"\n{title}")
        print("=" * len(title))

        # Group parameters by TorchParameter instance
        for param_obj in self.torch_params:
            print(f"\n{param_obj.param_name}:")
            print("-" * (len(param_obj.param_name) + 1))

            # Get all parameters for this TorchParameter
            param_items = list(param_obj.named_parameters())

            # Print raw parameter values
            for name, param in param_items:
                print(f"  {name}: {param.item():.6f}")

            # If this is a time parameter, also show actual time value
            try:
                actual_value = param_obj(0)  # Try calling with t=0
                if isinstance(actual_value, torch.Tensor):
                    actual_value = actual_value.item()
                    print(f"  → evaluated time (soft-plus): {actual_value:.6f}")
            except:
                pass  # Not a time parameter or doesn't support direct calling

            print()  # Add blank line between parameters


class XYAntiSymmetricProgram(FloquetProgram):
    """
    Example implementation of a Floquet program for a specific quantum system.
    """
    def __init__(self, num_sites: int = 8, spin='1/2', pbc=False,device: Optional[str] = None):
        self.num_sites = num_sites
        self.spin = spin
        self.pbc = pbc
        super().__init__(num_sites, spin, device)

    def _build_native_graph(self) -> TorchLatticeGraph:
        """Build the native Hamiltonian including the control terms."""
        # Create parameters for the control pulses
        DM_z = TorchParameter({'phase_factor': torch.pi/2, 'name': 'DM_phase'}, self.DM_z_period4_torch)
        XY_z = TorchParameter({'base_phase': torch.pi, 'phase_factor': 3*torch.pi/2, 'name': 'XY_phase'},
                              self.XY_z_period4_torch)
        native_off = TorchParameter({'coupling': 0.5, 'name': 'native_coupling'}, self.native_off_torch)

        # Define the interaction terms, native and control
        xy_terms = [['XX', native_off, 'nn'], ['YY', native_off, 'nn'],
                    ['z', DM_z, np.inf], ['z', XY_z, np.inf]]

        # Do not optimize the native coupling value
        self._set_param_optimization([native_off, DM_z, XY_z], optimize=False)

        return TorchLatticeGraph.from_torch_interactions(self.num_sites, xy_terms, pbc=self.pbc)

    def _build_target_graph(self) -> TorchLatticeGraph:
        """Build the target graph."""
        dm_terms = [['XY', 1/2, 'nn'], ['YX', -1/2, 'nn']]

        return TorchLatticeGraph.from_torch_interactions(self.num_sites, dm_terms, pbc=self.pbc)

    def _build_time_parameters(self) -> List[TorchParameter]:
        """Build the free evolution time parameters for the Floquet sequence."""
        # Create time parameters with positive constraint using softplus
        t_posJ = TorchParameter(
            {'value': -0.1, 'name': 'positive_J_time'},
            self._time_param
        )
        t_DM = TorchParameter(
            {'value': 1.1, 'name': 'DM_time'},
            self._time_param
        )
        t_negJ = TorchParameter(
            {'value': 0.9, 'name': 'negative_J_time'},
            self._time_param
        )

        return [t_posJ, t_DM, t_negJ]

    def _build_sequence_timings(self) -> tuple[List[str], List[TorchParameter]]:
        """Build the sequence timings."""
        # Currently, I am only interested in optimizing Floquet sequences with a known structure,
        # so _build_sequence_timings() should initialize self._hamiltonian_sequence so that we only
        # call _build_hamiltonian once. If the sequence of applied Hamiltonians needs to be mutable,
        # then it will be necessary to make _build_hamiltonian more efficient.

        # Get time parameters
        t_posJ, t_DM, t_negJ = self._build_time_parameters()

        # Define pulse sequence and timings
        pulse_list = ["nat", "+DM", "nat", "+XY", "nat", "-XY", "nat", "-DM", "nat"]
        dt_list = [t_posJ, 0, t_DM, 0, t_negJ, 0, t_DM, 0, t_posJ]

        self._evolve_times = dt_list
        self._pulse_parameters = pulse_list

    def DM_z_period4_torch(self, t, i, phase_factor=torch.tensor(torch.pi/2, dtype=torch.float64)):
        """DM interaction with period 4 pattern."""
        phase = phase_factor * (i % 4)
        if t == "+DM":
            return phase
        elif t == "-DM":
            return -phase
        else:
            return 0.

    def XY_z_period4_torch(self, t, i, base_phase=torch.tensor(torch.pi, dtype=torch.float64),
                        phase_factor=torch.tensor(3*torch.pi/2, dtype=torch.float64)):
        """XY interaction with period 4 pattern."""
        phase = base_phase - phase_factor * (i % 4)
        if t == "+XY":
            return phase
        elif t == "-XY":
            return -phase
        else:
            return 0.

    def native_off_torch(self, t, i, j, coupling=torch.tensor(0.5, dtype=torch.float64)):
        """Native interaction strength."""
        if t in ["+DM", "-DM", "+XY", "-XY"]:
            return 0.0
        else:
            return coupling
