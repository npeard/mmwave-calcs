import numpy as np
from itertools import product
from arc import DynamicPolarizability, ShirleyMethod
from arc import Cesium as Cs
from models.utility import wavelength2freq, power2field
from scipy.constants import c as C_c
from scipy.constants import epsilon_0


class ACStarkShift:
    def __init__(self, laserWaist=1e-6, n=6, l=0, j=0.5, mj=0.5, q=0):
        """
        Initialize the ACStarkShift object for computing AC Stark shifts given a
        state.

        Parameters
        ----------
        laserWaist : float, optional
            The waist of the laser beam in meters. Defaults to 1e-6.
        n : int, optional
            Principal quantum number of the state. Defaults to 6.
        l : int, optional
            Orbital angular momentum quantum number of the state. Defaults to 0.
        j : float, optional
            Total angular momentum quantum number of the state. Defaults to 0.5.
        mj : float, optional
            Magnetic quantum number of the state. Defaults to 0.5.
        q : int, optional
            Polarization of the laser. Defaults to 0.
        """
        self.laserWaist = laserWaist
        self.n = n
        self.l = l
        self.j = j
        self.mj = mj
        self.q = q
        self.atom = Cs()

        # calculation specific parameters
        self.target_state = [self.n, self.l, self.j]
        self.lmax = 3
        # for optical transitions, might want to adjust n_min and n_max such
        # that the transition is included in the calculation
        self.basis_n_min = self.n - 5
        self.basis_n_max = self.n + 5

    def ac_stark_shift_polarizability(self, wavelengthList, P):
        """
        Computes the AC Stark shift via the dynamic polarizability method.

        Parameters
        ----------
        wavelengthList : list
            A list of wavelengths to compute the AC Stark shift at.
        P : float
            The power of the laser.

        Returns
        -------
        U_AC : list
            A list of the AC Stark shifts at each of the wavelengths in
            wavelengthList.
        """
        calc = DynamicPolarizability(self.atom, *self.target_state)
        calc.defineBasis(self.basis_n_min, self.basis_n_max)
        alpha0_lst = []
        for wavelength in wavelengthList:
            alpha0, alpha1, alpha2, alphaC, alphaP, closestState = calc.getPolarizability(
                wavelength, units="SI")
            alpha0_lst.append(alpha0)

        I = 2 * P / (np.pi * self.laserWaist**2)
        U_AC = -np.real(alpha0_lst) * I / (2 * C_c * epsilon_0)

        return U_AC

    def ac_stark_shift_shirley(self, wavelengths, powers):

        """
        Computes the AC Stark shift using the Shirley method.

        Parameters
        ----------
        wavelengths : list
            A list of wavelengths to compute the AC Stark shift at.
        powers : list
            A list of powers to compute the AC Stark shift at.

        Returns
        -------
        u_shirley : array_like
            A 2D array of the AC Stark shifts at each combination of powers and
            wavelengths.
        """
        # Define the frequencies and electric field magnitudes that generate
        # the shifts
        freqs = wavelength2freq(wavelengths)
        eFields = power2field(powers, self.laserWaist)

        calc_full = ShirleyMethod(self.atom)

        # define a basis set that includes the local (microwave) and coupled
        # (optical) basis set
        calc_full.defineBasis(
            *self.target_state, self.mj, self.q, self.basis_n_min,
            self.basis_n_max,
            self.lmax,
            edN=0,
            progressOutput=False)
        basis_local = calc_full.basisStates
        calc_full.defineBasis(*self.target_state, self.mj, self.q, 6,
            10,
            self.lmax,
            edN=0,
            progressOutput=False)
        ryd_basis_coupled = calc_full.basisStates

        full_basis = basis_local
        if self.target_state[0] > 10:
            full_basis = basis_local + ryd_basis_coupled
            # If our target state is highly-excited and we are looking at AC
            # stark shifts near optical transitions, we need to include basis
            # states near the ground state that are coupled by the optical
            # transition, as well as those that are coupled by microwave
            # transitions.
        # Define the final basis
        calc_full.defineBasis(*self.target_state, self.mj, self.q,
                              basisStates=full_basis,
                              progressOutput=False)

        # Define the Hamiltonian
        calc_full.defineShirleyHamiltonian(fn=1)

        # Compute the AC Stark shifts
        u_shirley = []
        for freq, eField in product(freqs, eFields):
            calc_full.diagonalise(eField, freq, progressOutput=False)
            u_shirley.append(calc_full.targetShifts)
        results_Shirley = np.array(u_shirley).reshape((len(eFields),
                                                          len(freqs)))
        return results_Shirley
