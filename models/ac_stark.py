import numpy as np
from itertools import product
from arc import DynamicPolarizability, ShirleyMethod
from arc import Cesium as Cs
from models.utility import wavelength2freq, power2field
from scipy.constants import c as C_c
from scipy.constants import epsilon_0


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
