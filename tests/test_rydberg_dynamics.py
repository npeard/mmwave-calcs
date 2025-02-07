import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rydberg_dynamics import UnitaryRydberg, LossyRydberg


@pytest.fixture
def simulation_params():
    """Fixture providing common simulation parameters."""
    return {
        'duration': 5e-9,
        'delay': 10e-9,
        'hold': 27e-9,
        'probe_peak_power': 1e-3,
        'couple_power': 0.2,
        'Delta': 3.3e9
    }


@pytest.fixture
def unitary_runner():
    """Fixture providing a UnitaryRydberg instance."""
    return UnitaryRydberg()


@pytest.fixture
def lossy_runner():
    """
    Fixture providing a LossyRydberg instance with loss parameters set to zero.
    This makes the Lindblad evolution equivalent to the other evolution methods.
    """
    runner = LossyRydberg()
    runner.gamma2 = 0.0  # Set intermediate state loss to zero
    runner.gamma3 = 0.0  # Set Rydberg state loss to zero
    return runner


def test_evolution_methods_consistency(simulation_params, unitary_runner, lossy_runner):
    """
    Test that different evolution methods (unitary, Neumann, Lindblad) produce
    consistent results for identical input parameters.

    This test verifies that:
    1. Time arrays are identical across all methods
    2. Pulse shapes are identical across all methods
    3. Ground state populations match across all methods
    4. Excited state populations match across all methods
    5. Rydberg state populations match across all methods
    6. Loss population is zero in Lindblad evolution
    """
    # Get outputs from unitary evolution
    G1, E1, R1, pulse1, time1 = unitary_runner.probe_pulse_unitary(
        **simulation_params
    )

    # Get outputs from Neumann evolution
    G2, E2, R2, pulse2, time2 = unitary_runner.probe_pulse_neumann(
        **simulation_params
    )

    # Get outputs from Lindblad evolution
    G3, E3, R3, pulse3, time3, Loss = lossy_runner.probe_pulse_lindblad(
        **simulation_params
    )

    # Test time arrays
    np.testing.assert_allclose(
        time1, time2,
        rtol=1e-9,
        err_msg="Time arrays don't match between unitary and Neumann evolution"
    )
    np.testing.assert_allclose(
        time1, time3,
        rtol=1e-9,
        err_msg="Time arrays don't match between unitary and Lindblad evolution"
    )

    # Test pulse shapes
    np.testing.assert_allclose(
        pulse1, pulse2,
        rtol=1e-9,
        err_msg="Pulse shapes don't match between unitary and Neumann evolution"
    )
    np.testing.assert_allclose(
        pulse1, pulse3,
        rtol=1e-9,
        err_msg="Pulse shapes don't match between unitary and Lindblad evolution"
    )

    # Test ground state populations
    np.testing.assert_allclose(
        G1, G2,
        rtol=1e-2,
        err_msg="Ground state populations don't match between unitary and Neumann evolution"
    )
    # I think it makes sense that unitary and von Neumann evolution only roughly match.
    # The unitary evolution suffers from discretization errors and is likely less accurate.
    np.testing.assert_allclose(
        G2, G3,
        rtol=1e-3,
        err_msg="Ground state populations don't match between von Neumann and lossless Lindblad evolution"
    )

    # Test excited state populations
    np.testing.assert_allclose(
        E1, E2,
        rtol=1e-2,
        err_msg="Excited state populations don't match between unitary and Neumann evolution"
    )
    # I think it makes sense that unitary and von Neumann evolution only roughly match.
    # The unitary evolution suffers from discretization errors and is likely less accurate.
    np.testing.assert_allclose(
        E2, E3,
        rtol=1e-3,
        err_msg="Excited state populations don't match between von Neumann and lossless Lindblad evolution"
    )

    # Test Rydberg state populations
    np.testing.assert_allclose(
        R1, R2,
        rtol=1e-2,
        err_msg="Rydberg state populations don't match between unitary and Neumann evolution"
    )
    # I think it makes sense that unitary and von Neumann evolution only roughly match.
    # The unitary evolution suffers from discretization errors and is likely less accurate.
    np.testing.assert_allclose(
        R2, R3,
        rtol=1e-3,
        err_msg="Rydberg state populations don't match between von Neumann and lossless Lindblad evolution"
    )

    # Test that Loss is zero in Lindblad evolution
    np.testing.assert_allclose(
        Loss, np.zeros_like(Loss),
        rtol=1e-9,
        err_msg="Loss population is non-zero in Lindblad evolution"
    )
