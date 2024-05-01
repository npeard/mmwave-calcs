import models.rydberg_calcs as ryd
import models.pulse_calcs as pulses
import scipy.linalg
from scipy.integrate import solve_ivp
import numpy as np
from numba import njit, objmode


class UnitaryRydberg:
    def __init__(self):
        # intial state
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
        rho = np.reshape(rho.copy(), (3, 3))
        Ht = H.astype(np.complex128)

        return np.ravel(-1j * (Ht @ rho - rho @ Ht))

    def get_dot_rho(self, t, rho, duration, delay, hold, probe_peak_power,
                    Omega23):
        probe_power = (pulses.get_BlackmanPulse(t, duration, delay, hold) *
                       probe_peak_power)
        Omega12 = self.func_Omega12_from_Power(probe_power).item()
        Ht = self.get_hamiltonian(Omega12, Omega23, self.Delta,
                                  self.delta)

        return self.compute_dot_rho(rho, Ht)

    @staticmethod
    @njit('Tuple((float64[:], float64[:], float64[:]))(float64[:,:,:], '
          'float64, complex128[:])')
    def evolve_state(H_array, deltaT, psi0):
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
        # define time vector
        max_Omega12 = self.func_Omega12_from_Power(probe_peak_power)
        max_Omega23 = self.func_Omega23_from_Power(couple_power)
        if Delta is None:
            self.Delta = self.transition.get_OptimalDetuning(
                rabiFreq1=max_Omega12, rabiFreq2=max_Omega23)
        else:
            self.Delta = Delta
        max_freq = np.max([max_Omega12, max_Omega23, self.Delta])
        stop_time = delay + duration + hold + 10e-9
        self.time_array = np.linspace(0, stop_time, int(2 * stop_time *
                                                        max_freq) + 1)

        # define the pulse
        self.couple_power = couple_power
        self.probe_power = pulses.get_VectorizedBlackmanPulse(self.time_array,
                                                              duration, delay,
                                                              hold) * probe_peak_power
        self.probe_power[self.probe_power < 0] = 0
        Omega12_array = self.func_Omega12_from_Power(self.probe_power)
        Omega23 = self.func_Omega23_from_Power(self.couple_power).item()

        # compensate AC stark shift
        self.delta = self.transition.get_DiffRydACStark(probe_peak_power, couple_power)

        # calculate Hamiltonians
        H_array = self.get_hamiltonian_array(
            Omega12=Omega12_array, Omega23=Omega23, Delta=self.Delta,
            delta=self.delta)

        deltaT = self.time_array[1] - self.time_array[0]

        G_pop, E_pop, R_pop = self.evolve_state(H_array, deltaT, self.psi0)

        return G_pop, E_pop, R_pop, self.probe_power, self.time_array

    def probe_pulse_neumann(self, duration, delay, hold, probe_peak_power=10e-3,
                            couple_power=1, Delta=None):
        # define time vector
        max_Omega12 = self.func_Omega12_from_Power(probe_peak_power)
        max_Omega23 = self.func_Omega23_from_Power(couple_power)
        if Delta is None:
            self.Delta = self.transition.get_OptimalDetuning(
                rabiFreq1=max_Omega12, rabiFreq2=max_Omega23)
        else:
            self.Delta = Delta
        max_freq = np.max([max_Omega12, max_Omega23, self.Delta])
        stop_time = delay + duration + hold + 10e-9
        self.time_array = np.linspace(0, stop_time, int(2 * stop_time *
                                                        max_freq) + 1)

        # define the pulse
        self.couple_power = couple_power
        self.probe_power = pulses.get_VectorizedBlackmanPulse(self.time_array,
                                                              duration, delay,
                                                              hold) * probe_peak_power
        self.probe_power[self.probe_power < 0] = 0
        Omega12_array = self.func_Omega12_from_Power(self.probe_power)
        Omega23 = self.func_Omega23_from_Power(self.couple_power).item()

        # compensate AC stark shift
        self.delta = self.transition.get_DiffRydACStark(probe_peak_power,
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
        UnitaryRydberg.__init__(self)
        self.rho0 = np.asarray([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.complex128)

        # decay rates
        self.gamma2 = self.transition.get_E_Linewidth()
        self.gamma3 = self.transition.get_R_Linewidth()

    @staticmethod
    @njit('float64[:,:](float64, float64, float64, float64)')
    def get_hamiltonian(Omega12, Omega23, Delta, delta):
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
        probe_power = (pulses.get_BlackmanPulse(t, duration, delay, hold) *
                       probe_peak_power)
        Omega12 = self.func_Omega12_from_Power(probe_power).item()
        Ht = self.get_hamiltonian(Omega12, Omega23, self.Delta,
                                  self.delta)

        return self.compute_dot_rho(rho, Ht, self.gamma2, self.gamma3)

    def probe_pulse_lindblad(self, duration, delay, hold,
                             probe_peak_power, couple_power, Delta=None,
                             evolve_time = 0):
        # define time vector
        max_Omega12 = self.func_Omega12_from_Power(probe_peak_power)
        max_Omega23 = self.func_Omega23_from_Power(couple_power)
        if Delta is None:
            self.Delta = self.transition.get_OptimalDetuning(
                rabiFreq1=max_Omega12, rabiFreq2=max_Omega23)
        else:
            self.Delta = Delta
        max_freq = np.max([max_Omega12, max_Omega23, self.Delta])
        stop_time = delay + duration + hold + 10e-9 + evolve_time
        self.time_array = np.linspace(0, stop_time, int(2 * stop_time *
                                                        max_freq) + 1)

        # define the pulse
        self.couple_power = couple_power
        self.probe_power = pulses.get_VectorizedBlackmanPulse(self.time_array,
                                                              duration, delay,
                                                              hold) * probe_peak_power
        self.probe_power[self.probe_power < 0] = 0
        Omega23 = self.func_Omega23_from_Power(self.couple_power).item()

        # compensate AC stark shift
        self.delta = self.transition.get_DiffRydACStark(probe_peak_power,
                                                        couple_power)

        # solve initial value problem
        sol = solve_ivp(self.get_dot_rho, y0=np.ravel(self.rho0),
                        t_span=[np.min(self.time_array), np.max(
                            self.time_array)], t_eval=self.time_array, args=(
            duration, delay, hold, probe_peak_power, Omega23))
        rho_t = np.reshape(sol.y, (4, 4, len(self.time_array)))
        rho_t = np.real(rho_t)

        # populations
        G_pop = rho_t[0, 0, :]
        E_pop = rho_t[1, 1, :]
        R_pop = rho_t[2, 2, :]
        loss_pop = rho_t[3, 3, :]

        return G_pop, E_pop, R_pop, self.probe_power, self.time_array, loss_pop
