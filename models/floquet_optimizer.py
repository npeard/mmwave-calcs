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
import numpy as np
from typing import Dict, List, Any, Union, Optional, Callable
from models.floquet_program import FloquetProgram, XYAntiSymmetricProgram
from models.torch_chain import TorchParameter

class FloquetOptimizer:
    """
    Class for optimizing Floquet sequences using PyTorch.
    """
    def __init__(self, program: FloquetProgram, device: Optional[str] = None):
        """
        Initialize FloquetOptimizer.

        Parameters
        ----------
        program : FloquetProgram
            The Floquet program defining the native Hamiltonian and target evolution
        device : str, optional
            Device to run computations on ('cuda' or 'cpu'). If None, will use CUDA if available.
        """
        self.program = program
        self.optimizer = None
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_floquet_unitary(self, dt_list: List[float], hamiltonians: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute Floquet unitary evolution operator for a sequence of Hamiltonians and time steps.

        Parameters
        ----------
        dt_list : List[float]
            List of time steps
        hamiltonians : List[torch.Tensor]
            List of dense Hamiltonian matrices

        Returns
        -------
        torch.Tensor
            The Floquet unitary evolution operator
        """
        # Initialize identity matrix
        dim = hamiltonians[0].shape[0]
        U = torch.eye(dim, dtype=torch.complex128, device=self.device)

        # Process each time step sequentially
        for dt, H in zip(dt_list, hamiltonians):
            # H is already dense and on the correct device from FloquetProgram
            evolution = torch.matrix_exp(-1j * dt * H)
            U = evolution @ U

        return U

    def compute_fidelity_loss(self, U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
        """
        Compute fidelity loss between two unitary operators.

        Parameters
        ----------
        U1, U2 : torch.Tensor
            Unitary operators to compare

        Returns
        -------
        torch.Tensor
            Fidelity loss between the operators (0 = perfect match)
        """
        # Move to device if needed
        U1 = U1.to(self.device)
        U2 = U2.to(self.device)

        # Compute U1^† @ U2
        overlap = torch.abs(torch.trace(U1.conj().T @ U2))

        # Compute norms
        norm1 = torch.sqrt(torch.abs(torch.trace(U1.conj().T @ U1)))
        norm2 = torch.sqrt(torch.abs(torch.trace(U2.conj().T @ U2)))

        return torch.real(1 - overlap / (norm1 * norm2))

    def setup_optimizer(self, optimizer_class=torch.optim.Adam, **optimizer_kwargs):
        """Set up the PyTorch optimizer."""
        parameters = []
        for torch_param in self.program.torch_params:
            parameters.extend(torch_param.parameters())
        self.optimizer = optimizer_class(parameters, **optimizer_kwargs)

    def optimize_floquet_sequence(self, num_epochs: int = 100, verbose: bool = True) -> List[float]:
        """
        Optimize the Floquet sequence parameters to match a target unitary.

        Parameters
        ----------
        num_epochs : int, optional
            Number of optimization steps
        verbose : bool, optional
            Whether to print progress

        Returns
        -------
        List[float]
            List of loss values during optimization
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set up. Call setup_optimizer first.")

        losses = []

        # Print initial parameters
        self.program.print_parameters("Initial Parameters")

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            # Get the updated Floquet sequence
            delta_t, hamiltonian_list = self.program.get_floquet_sequence()

            # Evaluate time parameters
            dt_list = []
            for dt in delta_t:
                if isinstance(dt, TorchParameter):
                    dt_list.append(dt(0))
                elif callable(dt):
                    dt_list.append(dt())
                else:
                    dt_list.append(dt)

            floquet_period = sum(dt_list)
            target_U = self.program.get_target_unitary(floquet_period)

            # Get current unitary
            U = self.compute_floquet_unitary(dt_list, hamiltonian_list)

            # Compute loss
            loss = self.compute_fidelity_loss(U, target_U)
            losses.append(float(loss))

            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.6f}')

            # Backpropagate and optimize
            loss.backward()
            self.optimizer.step()

        # Print final parameters
        self.program.print_parameters("Final Parameters")

        return losses

if __name__ == "__main__":
    # Example usage
    import time
    import cProfile
    start_time = time.time()

    floq_opt = FloquetOptimizer(XYAntiSymmetricProgram(num_sites=4, pbc=True, device='cpu'), device='cpu')
    floq_opt.setup_optimizer(lr=0.001)
    floq_opt.optimize_floquet_sequence(num_epochs=200)
    #cProfile.run("floq_opt.optimize_floquet_sequence(num_epochs=100)")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
