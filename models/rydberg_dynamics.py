import models.rydberg_calcs as ryd
import models.pulse_calcs as pulses
import scipy.linalg
from scipy.integrate import solve_ivp
import numpy as np
from numba import njit, objmode


class UnitaryRydberg:
    def __init__(self):
        """
        Initialize the UnitaryRydberg class with default parameters. This
        class computes dynamics of the Rydberg transition without dissipation.
        
        This constructor sets up the initial quantum state, time array, and
        power parameters for the Rydberg transition dynamics. It also
        initializes the RydbergTransition instance and prepares interpolated
        functions for fast Rabi frequency lookup based on given power.
        
        Attributes
        ----------
        psi0 : np.ndarray
            Initial quantum state vector.
        rho0 : np.ndarray
            Initial density matrix for the quantum state.
        time_array : np.ndarray
            Discretized time array for evolution.
        probe_power : float or None
            Peak power of the probe laser in watts.
        couple_power : float or None
            Power of the coupling laser in watts.
        Delta : float or None
            Detuning parameter for the E transition.
        delta : float or None
            Detuning parameter for the Rydberg state.
        transition : RydbergTransition
            Instance of RydbergTransition for handling transition parameters.
        func_Omega12_from_Power : callable
            Function to compute Rabi frequency for the E state from power.
        func_Omega23_from_Power : callable
            Function to compute Rabi frequency for the Rydberg state from power.
        """
        # initial state
        self.psi0 = np.asarray([1, 0, 0], dtype=np.complex128)
        self.rho0 = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                               dtype=np.complex128)

        # parameters
        self.time_array = np.linspace(0, 1e-3, 1000)  # seconds
        self.probe_power = None  # watts
        self.couple_power = None  # watts
        self.Delta = None
        self.delta = None
        self.transition = ryd.RydbergTransition()

        # interpolated Rabi angular frequencies for fast lookup
        self.func_Omega12_from_Power = (
            self.transition.RabiAngularFreq_1_from_Power)
        self.func_Omega23_from_Power = (
            self.transition.RabiAngularFreq_2_from_Power)

    @staticmethod
    @njit('float64[:,:,:](float64[:], float64, float64, float64)')
    def get_hamiltonian_array(Omega12, Omega23, Delta, delta):
        # basis
        """
        Compute Hamiltonian array for a given array of Rabi frequencies and
        detunings.

        Parameters
        ----------
        Omega12 : float64[:]
            Array of probe laser Rabi angular frequencies.
        Omega23 : float64
            Coupling laser Rabi angular frequency.
        Delta : float64
            Detuning parameter for the E state transition.
        delta : float64
            Detuning parameter for the Rydberg state transition.

        Returns
        -------
        H : float64[:,:,:]
            Array of Hamiltonians in the three-state basis. The first axis
            indexes the Hamiltonian at each time step.
        """
        sm1 = np.asarray([[0, 1, 0], [0, 0, 0], [0, 0, 0]])  # sigma minus
        sm2 = np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 0]])  # sigma minus
        Detune = np.asarray([[0, 0, 0], [0, -1, 0], [0, 0, 0]])
        detune = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, -1]])

        # Hamiltonian
        H12 = Omega12[:, np.newaxis, np.newaxis] / 2 * (sm1 + sm1.T)[np.newaxis, :, :]
        H23 = Omega23 / 2 * (sm2 + sm2.T)
        H23 = H23[np.newaxis, :, :]
        HDelta = 0 - Detune * Delta - detune * delta
        HDelta = HDelta[np.newaxis, :, :]

        return H12 + H23 + HDelta

    @staticmethod
    @njit('float64[:,:](float64, float64, float64, float64)')
    def get_hamiltonian(Omega12, Omega23, Delta, delta):
        # basis
        """
        Compute Hamiltonian for a given set of Rabi frequencies and detunings.

        Parameters
        ----------
        Omega12 : float64
            Probe laser Rabi angular frequency.
        Omega23 : float64
            Coupling laser Rabi angular frequency.
        Delta : float64
            Detuning parameter for the E state transition.
        delta : float64
            Detuning parameter for the Rydberg state transition.

        Returns
        -------
        H : float64[:,:]
            Hamiltonian in the three-state basis.

        Notes
        -----
        The Hamiltonian is given by

        H = -1/2 \* (Omega12 \* (|e><g| + |g><e|) + Omega23 \* (|r><e| + |e><r|))
             - Delta \* |e><e| - delta \* |r><r|

        where |g>, |e>, and |r> are the ground, excited, and Rydberg states,
        respectively.
        """
        sm1 = np.asarray([[0, 1, 0], [0, 0, 0], [0, 0, 0]])  # sigma minus
        sm2 = np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 0]])  # sigma minus
        Detune = np.asarray([[0, 0, 0], [0, -1, 0], [0, 0, 0]])
        detune = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, -1]])

        # Hamiltonian
        H12 = Omega12 / 2 * (sm1 + sm1.T)
        H23 = Omega23 / 2 * (sm2 + sm2.T)
        HDelta = 0 - Detune * Delta - detune * delta

        return H12 + H23 + HDelta

    @staticmethod
    @njit('complex128[:](complex128[:], float64[:,:])')
    def compute_dot_rho(rho, H):
        """
        Compute the time derivative of the density matrix `rho` for a given
        Hamiltonian `H`.
    
        This function computes the Liouvillian superoperator acting on the density
        matrix `rho`, which represents the quantum state of a three-level system.
        The time derivative is calculated using the commutator of the Hamiltonian
        `H` with the density matrix `rho`, which describes the unitary evolution
        of the quantum state according to the Schrödinger equation.
    
        Parameters
        ----------
        rho : complex128[:]
            Flattened density matrix of the quantum state, expected to be a 3x3
            matrix when reshaped.
        H : float64[:,:]
            Hamiltonian matrix in the three-state basis.
    
        Returns
        -------
        complex128[:]
            Flattened array representing the time derivative of the density
            matrix `rho`.
        """
        rho = np.reshape(rho.copy(), (3, 3))
        Ht = H.astype(np.complex128)

        return np.ravel(-1j * (Ht @ rho - rho @ Ht))

    def get_dot_rho(self, t, rho, duration, delay, hold, probe_peak_power,
                    Omega23):
        """
        Compute the time derivative of the density matrix `rho` for a given
        Hamiltonian `H` corresponding to a probe pulse with the given
        parameters.
    
        Parameters
        ----------
        t : float
            Time at which to evaluate the time derivative.
        rho : complex128[:]
            Flattened density matrix of the quantum state, expected to be a 3x3
            matrix when reshaped.
        duration : float
            Duration of the probe pulse.
        delay : float
            Delay before the probe pulse starts.
        hold : float
            Duration of the flat top of the probe pulse.
        probe_peak_power : float
            Maximum power of the probe pulse.
        Omega23 : float
            Rabi angular frequency of the coupling laser.
    
        Returns
        -------
        complex128[:]
            Flattened array representing the time derivative of the density
            matrix `rho`.
    
        Notes
        -----
        The time derivative is calculated using the commutator of the Hamiltonian
        `H` with the density matrix `rho`, which describes the unitary evolution
        of the quantum state according to the Schrödinger equation. This
        function wraps the `compute_dot_rho` function which is Numba-compiled.
        """
        probe_power = (pulses.get_blackman_pulse(t, duration, delay, hold) *
                       probe_peak_power)
        Omega12 = self.func_Omega12_from_Power(probe_power).item()
        Ht = self.get_hamiltonian(Omega12, Omega23, self.Delta,
                                  self.delta)

        return self.compute_dot_rho(rho, Ht)

    @staticmethod
    @njit('Tuple((float64[:], float64[:], float64[:]))(float64[:,:,:], '
          'float64, complex128[:])')
    def evolve_state(H_array, deltaT, psi0):
        """
        Evolves the quantum state over time using a given Hamiltonian array.

        Parameters
        ----------
        H_array : float64[:,:,:]
            A 3D array representing the Hamiltonian at each time step.
        deltaT : float64
            The time step for the evolution.
        psi0 : complex128[:]
            The initial state vector of the quantum system.

        Returns
        -------
        Tuple(float64[:], float64[:], float64[:])
            A tuple containing the ground, intermediate, and Rydberg state
            populations over time.

        Notes
        -----
        The method calculates the unitary evolution of the quantum state
        using the matrix exponential of the Hamiltonian. The state vector
        is updated at each time step by applying the unitary operator.
        """
        with objmode(U_array='complex128[:,:,:]'):
            U_array = scipy.linalg.expm(1j * H_array * deltaT)
        # This could be faster if used Numpy functions to approximate
        # matrix exponential.

        psi = psi0.astype(np.complex128)
        psi_array = np.empty((H_array.shape[0], psi.shape[0]),
                             dtype=np.complex128)
        for i, U in enumerate(U_array):
            # psi = np.matmul(U, psi)
            psi_array[i, :] = psi
            psi = np.ascontiguousarray(U) @ psi

        G_pop = np.abs(psi_array[:, 0])**2
        E_pop = np.abs(psi_array[:, 1])**2
        R_pop = np.abs(psi_array[:, 2])**2

        return (G_pop.astype(np.float64), E_pop.astype(np.float64),
                R_pop.astype(np.float64))

    def probe_pulse_unitary(self, duration, delay, hold, probe_peak_power=10e-3,
                            couple_power=1, Delta=None):
        """
        Simulate the evolution of a quantum state under a probe pulse using the
        Numba-compiled `evolve_state` function.

        This function calculates the ground, intermediate, and Rydberg state
        populations over time when a probe pulse is applied to the system. It
        constructs the Hamiltonian for the system based on the given parameters,
        compensates for the AC Stark shift, and evolves the state using unitary
        dynamics.

        Parameters
        ----------
        duration : float
            The duration of the probe pulse.
        delay : float
            The delay before the probe pulse starts.
        hold : float
            The duration of the flat top of the probe pulse.
        probe_peak_power : float, optional
            The peak power of the probe pulse, default is 10e-3 W.
        couple_power : float, optional
            The power of the coupling laser, default is 1 W.
        Delta : float, optional
            The detuning for the transition. If None, the optimal detuning is
            calculated.

        Returns
        -------
        Tuple of float64[:]
            - Ground state population over time.
            - Intermediate state population over time.
            - Rydberg state population over time.
            - Probe pulse power over time.
            - Time array used for the simulation.
        """
        max_Omega12 = self.func_Omega12_from_Power(probe_peak_power)
        max_Omega23 = self.func_Omega23_from_Power(couple_power)
        if Delta is None:
            self.Delta = self.transition.get_optimal_detuning(
                rabiFreq1=max_Omega12, rabiFreq2=max_Omega23)
        else:
            self.Delta = Delta
        max_freq = np.max([max_Omega12, max_Omega23, self.Delta])
        stop_time = delay + duration + hold + 10e-9
        self.time_array = np.linspace(0, stop_time, int(2 * stop_time *
                                                        max_freq) + 1)

        # define the pulse
        self.couple_power = couple_power
        self.probe_power = pulses.get_vectorized_blackman_pulse(self.time_array,
                                                                duration, delay,
                                                                hold) * probe_peak_power
        self.probe_power[self.probe_power < 0] = 0
        Omega12_array = self.func_Omega12_from_Power(self.probe_power)
        Omega23 = self.func_Omega23_from_Power(self.couple_power).item()

        # compensate AC stark shift
        self.delta = self.transition.get_diff_ryd_ac_stark(probe_peak_power, couple_power)

        # calculate Hamiltonians
        H_array = self.get_hamiltonian_array(
            Omega12=Omega12_array, Omega23=Omega23, Delta=self.Delta,
            delta=self.delta)

        deltaT = self.time_array[1] - self.time_array[0]

        G_pop, E_pop, R_pop = self.evolve_state(H_array, deltaT, self.psi0)

        return G_pop, E_pop, R_pop, self.probe_power, self.time_array

    def probe_pulse_neumann(self, duration, delay, hold, probe_peak_power=10e-3,
                            couple_power=1, Delta=None):
        """
        Simulate the evolution of a quantum state under a probe pulse using the
        von Neumann equation and solving as an initial value problem. Mainly
        used to compare to results from the unitary evolution function above.

        This function calculates the populations of the ground, intermediate, and
        Rydberg states over time when a probe pulse is applied to the system. The
        Hamiltonian is constructed based on the given parameters, and the state is
        evolved using the von Neumann equation.

        Parameters
        ----------
        duration : float
            Duration of the probe pulse.
        delay : float
            Delay before the probe pulse starts.
        hold : float
            Duration of the flat top of the probe pulse.
        probe_peak_power : float, optional
            Peak power of the probe pulse, default is 10e-3 W.
        couple_power : float, optional
            Power of the coupling laser, default is 1 W.
        Delta : float, optional
            Detuning for the transition. If None, the optimal detuning is
            calculated.

        Returns
        -------
        Tuple of float64[:]
            - Ground state population over time.
            - Intermediate state population over time.
            - Rydberg state population over time.
            - Probe pulse power over time.
            - Time array used for the simulation.
        """
        max_Omega12 = self.func_Omega12_from_Power(probe_peak_power)
        max_Omega23 = self.func_Omega23_from_Power(couple_power)
        if Delta is None:
            self.Delta = self.transition.get_optimal_detuning(
                rabiFreq1=max_Omega12, rabiFreq2=max_Omega23)
        else:
            self.Delta = Delta
        max_freq = np.max([max_Omega12, max_Omega23, self.Delta])
        stop_time = delay + duration + hold + 10e-9
        self.time_array = np.linspace(0, stop_time, int(2 * stop_time *
                                                        max_freq) + 1)

        # define the pulse
        self.couple_power = couple_power
        self.probe_power = pulses.get_vectorized_blackman_pulse(self.time_array,
                                                                duration, delay,
                                                                hold) * probe_peak_power
        self.probe_power[self.probe_power < 0] = 0
        Omega12_array = self.func_Omega12_from_Power(self.probe_power)
        Omega23 = self.func_Omega23_from_Power(self.couple_power).item()

        # compensate AC stark shift
        self.delta = self.transition.get_diff_ryd_ac_stark(probe_peak_power,
                                                           couple_power)

        # solve initial value problem
        sol = solve_ivp(self.get_dot_rho, y0=np.ravel(self.rho0),
                        t_span=[np.min(self.time_array), np.max(
                            self.time_array)], t_eval=self.time_array, args=(
            duration, delay, hold, probe_peak_power, Omega23))
        rho_t = np.reshape(sol.y, (3, 3, len(self.time_array)))
        rho_t = np.real(rho_t)

        # populations
        G_pop = rho_t[0, 0, :]
        E_pop = rho_t[1, 1, :]
        R_pop = rho_t[2, 2, :]

        return G_pop, E_pop, R_pop, self.probe_power, self.time_array


class LossyRydberg(UnitaryRydberg):
    def __init__(self):
        """
        Initialize a LossyRydberg instance which inherits from UnitaryRydberg
        and adds dissipation to the evolution.

        This constructor sets up the initial density matrix for the quantum
        system and retrieves the decay rates for the intermediate and Rydberg
        states.

        Attributes
        ----------
        rho0 : np.ndarray
            The initial density matrix of the system, represented as a 4x4
            complex matrix with the ground state population set to 1.
        gamma2 : float
            The linewidth of the intermediate state, in Hz, obtained from the
            transition properties.
        gamma3 : float
            The linewidth of the Rydberg state, in Hz, obtained from the
            transition properties.
        """
        UnitaryRydberg.__init__(self)
        self.rho0 = np.asarray([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.complex128)

        # decay rates
        self.gamma2 = self.transition.get_e_linewidth()
        self.gamma3 = self.transition.get_r_linewidth()

    @staticmethod
    @njit('float64[:,:](float64, float64, float64, float64)')
    def get_hamiltonian(Omega12, Omega23, Delta, delta):
        """
        Constructs the Hamiltonian matrix for a three-level quantum system.

        This function calculates the Hamiltonian for a three-level system
        based on the given Rabi frequencies and detunings. The Hamiltonian
        accounts for the couplings between the ground, intermediate, and
        Rydberg states, and the detunings for the transitions.

        Parameters
        ----------
        Omega12 : float
            Rabi frequency for the transition between the ground and
            intermediate state.
        Omega23 : float
            Rabi frequency for the transition between the intermediate and
            Rydberg state.
        Delta : float
            Detuning of the laser field for the intermediate state.
        delta : float
            Detuning of the laser field for the Rydberg state.

        Returns
        -------
        np.ndarray
            A 4x4 Hamiltonian matrix representing the dynamics of the system.
        """
        # basis
        sm1 = np.asarray([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])  # sigma minus
        sm2 = np.asarray([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])  # sigma minus
        Detune = np.asarray([[0, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        detune = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0]])

        # Hamiltonian
        H12 = Omega12 / 2 * (sm1 + sm1.T)
        H23 = Omega23 / 2 * (sm2 + sm2.T)
        HDelta = 0 - Detune * Delta - detune * delta

        return H12 + H23 + HDelta

    @staticmethod
    @njit('complex128[:](complex128[:], float64[:,:], float64, float64)')
    def compute_dot_rho(rho, H, gamma2, gamma3):
        """
        Compute the time derivative of the density matrix rho for a three-level
        quantum system using the von Neumann equation.

        Parameters
        ----------
        rho : np.ndarray
            A 4x4 density matrix representing the quantum state of the system
        H : np.ndarray
            A 4x4 Hamiltonian matrix representing the dynamics of the system
        gamma2 : float
            The linewidth of the intermediate state, in Hz
        gamma3 : float
            The linewidth of the Rydberg state, in Hz

        Returns
        -------
        np.ndarray
            The time derivative of the density matrix rho, a 4x4 complex matrix

        Notes
        -----
        The von Neumann equation is a quantum mechanical equation that describes
        the time evolution of a quantum system. The Liouville equation is a
        special case of the von Neumann equation for a closed system. The
        von Neumann equation is given by:

        d/dt rho = -i [H, rho] + gamma2 * L(spLoss1) + gamma3 * L(spLoss2)

        where rho is the density matrix of the system, H is the Hamiltonian
        matrix, gamma2 and gamma3 are the linewidths of the intermediate and
        Rydberg states, and L is the Lindblad superoperator.

        The Lindblad superoperator is given by:

        L(A) = A @ rho @ A.T - 0.5 * (A.T @ A @ rho + rho @ A.T @ A)

        The Lindblad superoperator is used to describe the loss of coherence
        due to the decay of the intermediate and Rydberg states. The loss
        channel is modeled as a decay from the Rydberg state to the ground
        state, and from the intermediate state to the ground state.
        """
        # loss channel
        spLoss1 = np.asarray(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.complex128)  # sigma plus, since the loss channel is "above" the Rydberg state
        spLoss2 = np.asarray(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
            dtype=np.complex128)  # sigma plus, since the loss channel is "above" the Rydberg state

        rho = np.reshape(rho.copy(), (4, 4))
        Ht = H.astype(np.complex128)

        vonNeumann = -1j * (Ht @ rho - rho @ Ht)

        loss_e = gamma2 * (spLoss1 @ (rho @ spLoss1.T)
                           - 0.5 * (spLoss1.T @ (spLoss1 @ rho) + rho @ (spLoss1.T @ spLoss1)))
        loss_r = gamma3 * (spLoss2 @ (rho @ spLoss2.T)
                           - 0.5 * (spLoss2.T @ (spLoss2 @ rho) + rho @ (spLoss2.T @ spLoss2)))

        dot_rho = vonNeumann + loss_e + loss_r

        return np.ravel(dot_rho)

    def get_dot_rho(self, t, rho, duration, delay, hold, probe_peak_power,
                    Omega23):
        """
        Compute the time derivative of the density matrix `rho` for a given
        Hamiltonian `H` corresponding to a probe pulse with the given
        parameters.

        Parameters
        ----------
        t : float
            Time at which to evaluate the time derivative.
        rho : complex128[:]
            Flattened density matrix of the quantum state, expected to be a 4x4
            matrix when reshaped.
        duration : float
            Duration of the probe pulse.
        delay : float
            Delay before the probe pulse starts.
        hold : float
            Duration of the flat top of the probe pulse.
        probe_peak_power : float
            Maximum power of the probe pulse.
        Omega23 : float
            Rabi angular frequency of the coupling laser.

        Returns
        -------
        complex128[:]
            Flattened array representing the time derivative of the density
            matrix `rho`.

        Notes
        -----
        The time derivative is calculated using the commutator of the
        Hamiltonian `H` with the density matrix `rho`, which describes the
        unitary evolution of the quantum state according to the Schrödinger
        equation. This function wraps the `compute_dot_rho` function which is
        Numba-compiled.
        """
        probe_power = (pulses.get_blackman_pulse(t, duration, delay, hold) *
                       probe_peak_power)
        Omega12 = self.func_Omega12_from_Power(probe_power).item()
        Ht = self.get_hamiltonian(Omega12, Omega23, self.Delta, self.delta)

        return self.compute_dot_rho(rho, Ht, self.gamma2, self.gamma3)

    def get_dot_rho_duo(self, t, rho, probe_duration, probe_delay, probe_hold,
                        probe_peak_power, couple_duration, couple_delay,
                        couple_hold, couple_peak_power):
        """
        Compute the time derivative of the density matrix `rho` for a given
        Hamiltonian `H` corresponding to a dual probe and coupling pulse with
        the given parameters.
    
        Parameters
        ----------
        t : float
            Time at which to evaluate the time derivative.
        rho : complex128[:]
            Flattened density matrix of the quantum state, expected to be a 4x4
            matrix when reshaped.
        probe_duration : float
            Duration of the probe pulse.
        probe_delay : float
            Delay before the probe pulse starts.
        probe_hold : float
            Duration of the flat top of the probe pulse.
        probe_peak_power : float
            Maximum power of the probe pulse.
        couple_duration : float
            Duration of the coupling pulse.
        couple_delay : float
            Delay before the coupling pulse starts.
        couple_hold : float
            Duration of the flat top of the coupling pulse.
        couple_peak_power : float
            Maximum power of the coupling pulse.
    
        Returns
        -------
        complex128[:]
            Flattened array representing the time derivative of the density
            matrix `rho`.
    
        Notes
        -----
        The time derivative is calculated using the commutator of the Hamiltonian
        `H` with the density matrix `rho`, which describes the unitary evolution
        of the quantum state according to the Schrödinger equation. This function
        wraps the `compute_dot_rho` function which is Numba-compiled.
        """
        probe_power = (pulses.get_blackman_pulse(t, probe_duration,
                                                 probe_delay, probe_hold) *
                       probe_peak_power)
        Omega12 = self.func_Omega12_from_Power(probe_power).item()
        couple_power = (pulses.get_blackman_pulse(t, couple_duration,
                                                  couple_delay, couple_hold) *
                        couple_peak_power)
        Omega23 = self.func_Omega23_from_Power(couple_power).item()
        Ht = self.get_hamiltonian(Omega12, Omega23, self.Delta, self.delta)

        return self.compute_dot_rho(rho, Ht, self.gamma2, self.gamma3)

    def probe_pulse_lindblad(self, duration, delay, hold,
                             probe_peak_power, couple_power, Delta=None,
                             evolve_time=0):
        """
        Solve the Lindblad master equation for a probe pulse.

        Parameters
        ----------
        duration : float
            Duration of the probe pulse.
        delay : float
            Delay before the probe pulse starts.
        hold : float
            Duration of the flat top of the probe pulse.
        probe_peak_power : float
            Maximum power of the probe pulse.
        couple_power : float
            Power of the coupling pulse.
        Delta : float, optional
            Detuning from the intermediate state. If `None`, the optimal detuning
            is calculated using `transition.get_optimal_detuning`.
        evolve_time : float, optional
            Additional time to evolve the system after the probe pulse.

        Returns
        -------
        G_pop : real128[:]
            Population of the ground state.
        E_pop : real128[:]
            Population of the intermediate state.
        R_pop : real128[:]
            Population of the Rydberg state.
        probe_power : real128[:]
            Power of the probe pulse as a function of time.
        time_array : real128[:]
            Time array.
        loss_pop : real128[:]
            Population lost to other states.
        """
        max_Omega12 = self.func_Omega12_from_Power(probe_peak_power)
        max_Omega23 = self.func_Omega23_from_Power(couple_power)
        if Delta is None:
            self.Delta = self.transition.get_optimal_detuning(
                rabiFreq1=max_Omega12, rabiFreq2=max_Omega23)
        else:
            self.Delta = Delta
        max_freq = np.max([max_Omega12, max_Omega23, self.Delta])
        stop_time = delay + duration + hold + 10e-9 + evolve_time
        self.time_array = np.linspace(0, stop_time, int(2 * stop_time *
                                                        max_freq) + 1)

        # define the pulse
        self.couple_power = couple_power
        self.probe_power = pulses.get_vectorized_blackman_pulse(self.time_array,
                                                                duration, delay,
                                                                hold) * probe_peak_power
        self.probe_power[self.probe_power < 0] = 0
        Omega23 = self.func_Omega23_from_Power(self.couple_power).item()

        # compensate AC stark shift
        self.delta = self.transition.get_diff_ryd_ac_stark(probe_peak_power,
                                                           couple_power)

        # solve initial value problem
        sol = solve_ivp(self.get_dot_rho, y0=np.ravel(self.rho0),
                        t_span=[np.min(self.time_array), np.max(
                            self.time_array)], t_eval=self.time_array, args=(
            duration, delay, hold, probe_peak_power, Omega23), first_step=1e-9)
        rho_t = np.reshape(sol.y, (4, 4, len(self.time_array)))
        rho_t = np.real(rho_t)

        # populations
        G_pop = rho_t[0, 0, :]
        E_pop = rho_t[1, 1, :]
        R_pop = rho_t[2, 2, :]
        loss_pop = rho_t[3, 3, :]

        return G_pop, E_pop, R_pop, self.probe_power, self.time_array, loss_pop

    def duo_pulse_lindblad(self, probe_duration, probe_delay, probe_hold,
                           probe_peak_power, couple_duration, couple_delay,
                           couple_hold, couple_peak_power,
                           Delta=0.0):
        """
        Simulate a two-pulse sequence, first a probe pulse, then a couple pulse,
        and compute the populations of the ground state, the intermediate state,
        and the Rydberg state, as well as the probe and couple pulse powers.

        Parameters
        ----------
        probe_duration : float
            Duration of the probe pulse in seconds.
        probe_delay : float
            Delay between the start of the simulation and the start of the
            probe pulse in seconds.
        probe_hold : float
            Duration of the hold period after the probe pulse in seconds.
        probe_peak_power : float
            Peak power of the probe pulse in Watts.
        couple_duration : float
            Duration of the couple pulse in seconds.
        couple_delay : float
            Delay between the start of the simulation and the start of the
            couple pulse in seconds.
        couple_hold : float
            Duration of the hold period after the couple pulse in seconds.
        couple_peak_power : float
            Peak power of the couple pulse in Watts.
        Delta : float, optional
            Detuning of the probe pulse in Hz. Defaults to 0.

        Returns
        -------
        G_pop : numpy.ndarray
            Population of the ground state.
        E_pop : numpy.ndarray
            Population of the intermediate state.
        R_pop : numpy.ndarray
            Population of the Rydberg state.
        probe_power : numpy.ndarray
            Probe power at each time step.
        couple_power : numpy.ndarray
            Couple power at each time step.
        time_array : numpy.ndarray
            Time array for the simulation.
        loss_pop : numpy.ndarray
            Population of the loss state.
        """
        max_Omega12 = self.func_Omega12_from_Power(probe_peak_power)
        max_Omega23 = self.func_Omega23_from_Power(couple_peak_power)
        self.Delta = Delta
        self.delta = 0.0
        max_freq = np.max([max_Omega12, max_Omega23, self.Delta])
        stop_time = (probe_delay + probe_duration + probe_hold +
                     couple_delay + couple_duration + couple_hold + 10e-9)
        self.time_array = np.linspace(0, stop_time, int(2 * stop_time *
                                                        max_freq) + 1)

        # define the pulse
        self.couple_power = (pulses.get_vectorized_blackman_pulse(
            self.time_array, couple_duration, couple_delay, couple_hold) *
            couple_peak_power)
        self.couple_power[self.couple_power < 0] = 0
        self.probe_power = pulses.get_vectorized_blackman_pulse(self.time_array,
                                                                probe_duration, probe_delay,
                                                                probe_hold) * probe_peak_power
        self.probe_power[self.probe_power < 0] = 0

        # solve initial value problem
        sol = solve_ivp(self.get_dot_rho_duo, y0=np.ravel(self.rho0),
                        t_span=[np.min(self.time_array), np.max(
                            self.time_array)], t_eval=self.time_array, args=(
            probe_duration, probe_delay, probe_hold, probe_peak_power,
            couple_duration, couple_delay, couple_hold,
            couple_peak_power), first_step=1e-9)
        rho_t = np.reshape(sol.y, (4, 4, len(self.time_array)))
        rho_t = np.real(rho_t)

        # populations
        G_pop = rho_t[0, 0, :]
        E_pop = rho_t[1, 1, :]
        R_pop = rho_t[2, 2, :]
        loss_pop = rho_t[3, 3, :]

        return (G_pop, E_pop, R_pop, self.probe_power, self.couple_power,
                self.time_array, loss_pop)
