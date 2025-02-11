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
        self._native_graph = self._build_native_graph()
        self._target_graph = self._build_target_graph()

        # Pulse program parameters, lists of pulse parameters (can be times or strings)
        # and associated evolution time for that pulse
        self._build_sequence_timings()
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
        raise NotImplementedError()

    def _build_hamiltonian(self, graph: TorchLatticeGraph, t: float) -> torch.Tensor:
        """Build Hamiltonian at time t."""
        # Get dimension from number of sites
        spin_op = TorchSpinOperator(self.num_sites, self.spin)
        dim = spin_op.dim ** self.num_sites  # dimension of full Hilbert space

        # Initialize Hamiltonian
        H = torch.zeros((dim, dim), dtype=torch.complex128, device=self.device)

        # Add all interaction terms
        for op_string, interactions in graph.interaction_dict.items():
            for interaction in interactions:
                strength = graph.evaluate_interaction(interaction, t)
                sites = interaction[1:]
                # Get sparse operator and convert to dense
                op = spin_op.get_operator_sparse(op_string, sites).to_dense()
                if isinstance(strength, torch.Tensor):
                    op = op * strength
                else:
                    op = op * float(strength)
                H = H + op

        return H

    def _build_native_hamiltonian(self, t):
        """
        Build the native Hamiltonian graph. This should include the time-dependent
        control terms.
        """
        return self._build_hamiltonian(self._native_graph, t=t)

    def _build_target_hamiltonian(self):
        """
        Build the target Hamiltonian graph. This Hamiltonian should have no
        time-dependence.
        """
        return self._build_hamiltonian(self._target_graph, t=0)

    def get_floquet_sequence(self):
        """
        Return the sequence of Hamiltonians and times for the Floquet system.
        This is the main function to be called by an optimizer, which performs the evolution
        computation.
        """
        hamiltonians = [self._build_native_hamiltonian(p) for p in self._pulse_parameters]
        delta_t = [dt if dt != 0 else 1 for dt in self._evolve_times]

        return delta_t, hamiltonians

    def get_target_unitary(self, floquet_period):
        """
        Compute the target unitary for the given Floquet period.
        Use this to compute loss metrics.
        """
        U = torch.matrix_exp(-1j * self._build_target_hamiltonian() * floquet_period).detach()
        return U
        #return torch.matrix_exp(-1j * self._build_target_hamiltonian() * floquet_period).detach()

    def _time_param(self, t, value=torch.tensor(0.1, dtype=torch.float64)):
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
        print("-" * len(title))

        # Group parameters by TorchParameter instance
        for i, param_obj in enumerate(self.torch_params):
            print(f"\nTorchParameter {i+1}:")

            # Get all parameters for this TorchParameter
            param_items = list(param_obj.named_parameters())

            # Print raw parameter values
            print("  Raw parameter values:")
            for name, param in param_items:
                print(f"    {name}: {param.item():.6f}")

            # If this is a time parameter, also show actual time value
            try:
                actual_value = param_obj(0)  # Try calling with t=0
                if isinstance(actual_value, torch.Tensor):
                    print("  Actual time value:")
                    print(f"    {actual_value.item():.6f}")
            except:
                pass  # Not a time parameter or doesn't support direct calling


class XYAntiSymmetricProgram(FloquetProgram):
    """
    Example implementation of a Floquet program for a specific quantum system.
    """
    def __init__(self, num_sites: int = 8, spin='1/2'):
        self.num_sites = num_sites
        self.spin = spin
        super().__init__(num_sites, spin)

    def _build_native_graph(self) -> TorchLatticeGraph:
        """Build the native Hamiltonian including the control terms."""
        # Create parameters for the control pulses
        DM_z = TorchParameter({'phase_factor': torch.pi/2}, self.DM_z_period4_torch)
        XY_z = TorchParameter({'base_phase': torch.pi, 'phase_factor': 3*torch.pi/2},
                              self.XY_z_period4_torch)
        native_off = TorchParameter({'coupling': 0.5}, self.native_off_torch)

        # Define the interaction terms, native and control
        xy_terms = [['XX', native_off, 'nn'], ['YY', native_off, 'nn'],
                    ['z', DM_z, np.inf], ['z', XY_z, np.inf]]

        # Do not optimize the native coupling value
        self._set_param_optimization([native_off], optimize=False)

        return TorchLatticeGraph.from_torch_interactions(self.num_sites, xy_terms)

    def _build_target_graph(self) -> TorchLatticeGraph:
        """Build the target graph."""
        dm_terms = [['XY', 1/2, 'nn'], ['YX', -1/2, 'nn']]

        return TorchLatticeGraph.from_torch_interactions(self.num_sites, dm_terms)

    def _build_time_parameters(self) -> List[TorchParameter]:
        """Build the free evolution time parameters for the Floquet sequence."""
        # Create time parameters with positive constraint using softplus
        t_posJ = TorchParameter({'value': torch.log(torch.tensor(0.1) + 1.0)}, self._time_param)
        t_DM = TorchParameter({'value': torch.log(torch.tensor(0.2) + 1.0)}, self._time_param)
        t_negJ = TorchParameter({'value': torch.log(torch.tensor(0.15) + 1.0)}, self._time_param)

        return [t_posJ, t_DM, t_negJ]

    def _build_sequence_timings(self) -> tuple[List[str], List[TorchParameter]]:
        """Build the sequence timings."""
        # Get time parameters
        t_posJ, t_DM, t_negJ = self._build_time_parameters()

        # Define pulse sequence and timings
        pulse_list = ["nat", "+DM", "nat", "+XY", "nat", "-XY", "nat", "-DM", "nat"]
        dt_list = [t_posJ, 0, t_DM, 0, t_negJ, 0, t_DM, 0, t_posJ]  # Now passing parameters instead of evaluated values

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
