import numpy as np
from itertools import product
from arc import DynamicPolarizability, ShirleyMethod
from arc import Cesium as Cs
from models.utility import wavelength2freq, power2field, p2i_gaussian, w2omega, E2f, E2T, Rayleigh_range, trap_frequency
from scipy.constants import c as C_c
from scipy.constants import epsilon_0
import scipy.constants as const
from models.rydberg_calcs import RydbergTransition, alkaline_constant


class ACStarkShift:
    """Initialize a state and compute AC Stark shifts for that state"""

    def __init__(self, laserWaist=1e-6, n=6, l=0, j=0.5, mj=0.5, q=0):
        self.laserWaist = laserWaist
        self.n = n
        self.l = l
        self.j = j
        self.mj = mj
        self.q = q
        self.atom = Cs()

        # calculation specific parameters
        self.state = [self.n, self.l, self.j]
        self.lmax = 2
        self.n_basis = 5

    def ac_stark_shift_polarizability(self, wavelengthList, P):
        calc = DynamicPolarizability(self.atom, *self.state)
        calc.defineBasis(self.atom.groundStateN, self.n_basis)
        alpha0_lst = []
        for wavelength in wavelengthList:
            alpha0, alpha1, alpha2, alphaC, alphaP, closestState = calc.getPolarizability(
                wavelength, units="SI")
            alpha0_lst.append(alpha0)

        I = 2 * P / (np.pi * self.laserWaist**2)
        U_AC = -np.real(alpha0_lst) * I / (2 * C_c * epsilon_0)

        return U_AC

    def ac_stark_shift_shirley(self, wavelengths, powers):

        freqs = wavelength2freq(wavelengths)

        eFields = power2field(powers, self.laserWaist)  # 0.1*1e3  # V/m

        calc_full = ShirleyMethod(self.atom)
        calc_full.defineBasis(
            *self.state, self.mj, self.q, self.state[0] - self.n_basis, self.state[0] +
            self.n_basis,
            self.lmax,
            edN=0,
            progressOutput=False
        )
        calc_full.defineShirleyHamiltonian(fn=1)

        U_AC_Shirley = []

        for freq, eField in product(freqs, eFields):
            calc_full.diagonalise(eField, freq, progressOutput=False)
            U_AC_Shirley.append(calc_full.targetShifts)
        results_Shirley = np.array(U_AC_Shirley).reshape((len(eFields),
                                                          len(freqs)))

        return results_Shirley
    
    def angular_freq_counter(self,wavelength1, wavelength2):
        """
        return the frequency difference between two wavelengths

        Parameters:
            wavelength1: wavelength 1 (m)
            wavelength2: wavelength 2 (m)
        """
        return w2omega(wavelength1) - w2omega(wavelength2)

    def angular_freq_co(sefl, wavelength1, wavelength2):
        """
        return the frequency difference between two wavelengths

        Parameters:
            wavelength1: wavelength 1 (m)
            wavelength2: wavelength 2 (m)
        """
        return w2omega(wavelength1) + w2omega(wavelength2)
    
    def dipole_trap_depth(self, Gamma, lamda0, lamda, I, I_sat, method = "far detuned", debug = False):
        """
        return the dipole trap potential using Eq 9.46 from Atomic physics (Foot)
        
        Parameters:
            Gamma: linewidth
            angular_detuning: frequency detuning between the dipole trap wavelength and the dominant transition
            I: intensity of dipole trap
            I_sat: saturate intensity
        """
        if method == "far detuned":
            U0 = -const.hbar*Gamma/8*Gamma/self.angular_freq_counter(lamda0,lamda)*I/I_sat
        elif method == "near detuned":
            U0 = -const.hbar*Gamma/8*Gamma*(1/self.angular_freq_counter(lamda0,lamda)+1/self.angular_freq_co(lamda0,lamda))*I/I_sat
        elif method == "multilevel":
            raise ValueError("Method to be implemented")
                #U0 = -const.hbar*Gamma/8*Gamma*I/I_sat*np.sum([1/angular_freq_counter(lamda0, lamda)-1/angular_freq_co(lamda0, lamda),])
        else:
            raise ValueError("Invalid method. Choose 'far detuned' or 'near detuned'.")
        
        if debug == True:
            print(f"U0 = {U0} J")
            print(f"Isat ratio = {I/I_sat/1e7}*10^7")
        return U0

    def dipole_trap_parameters(self, atom, w0, P, lamda_laser, q, method = "far detuned", excited_state_f = 5, debug = False):
        Ry= RydbergTransition()
        lamda_D2 = Ry.alkaline_wavelength_hfs(atom, transition = "D2", debug = debug) #852e-9 # D2 line
        ground_state = alkaline_constant(atom).ground_state #[6, 0, 0.5] # n, l, j for ground state
        ground_state_f = alkaline_constant(atom).ground_state_f # Hyper fine F = 3
        excited_state = alkaline_constant(atom).D2_state # n, l, j for 
        

        if q == 0:
            # linear polarization
            I_sat = alkaline_constant(atom).I_sat_D2_pi #W/m^2 1.6536(15) mW/cm2
        elif abs(q) == 1:
            # circular polarization
            I_sat = alkaline_constant(atom).I_sat_D2_sigma #W/m^2 1.6536(15) mW/cm2
        else:
            # isotropic polarization
            print("no such polarization")


        m = atom.mass #kg cesium mass

        Gamma = RydbergTransition(n2 = excited_state[0], l2 = excited_state[1], j2 = excited_state[2]).get_E_Linewidth()# linewidth D2
        # I_sat = I_sat_circular(Gamma, lamda_D2) # the circular polarized saturation intensity can be calculated instead of import from constantw 

        I = p2i_gaussian(P,w0)

        U0 = self.dipole_trap_depth(Gamma, lamda_D2, lamda_laser, I, I_sat, method=method, debug = debug)
        T = E2T(abs(U0), debug = debug)
        f_U0 = E2f(U0, debug = debug)
        zR = Rayleigh_range(w0, lamda_laser, debug = True)
        omega_H, omega_L = trap_frequency(abs(U0), w0, zR, m, debug = debug)

        if debug == True:
            print(f"mass = {m} kg")
            print(f"I_sat = {I_sat} W/m^2")
            omega_0 = w2omega(lamda_D2)
            omega = w2omega(lamda_laser)
            angular_detuning = omega - omega_0
            print(f"detuning ratio = {-angular_detuning/Gamma/1e7}*10^7")

        return U0, T, f_U0, zR, omega_H, omega_L
