# Calculate gate errors based on this paper:
# Jiang, X., Scott, J., Friesen, M. & Saffman, M. Sensitivity of quantum gate
# fidelity to laser phase and intensity noise. Phys. Rev. A 107, 042611 (2023).

import numpy as np
import models.rydberg_calcs as calcs


def whiteGateError(FWHM1, FWHM2, Pp, Pc, N = 1/2):
    # Calculate the gate error for a pi*N gate due to laser phase noise
    transition = calcs.RydbergTransition()
    rabiFreqTotal = transition.get_totalRabiAngularFreq(Pp, Pc) # in 2pi*Hz
    h1 = FWHM1/(2*np.pi)
    h2 = FWHM2/(2*np.pi)

    epsilon = np.pi**3 * (h1+h2)*N / rabiFreqTotal

    return epsilon


def servoGateError(sg, fg, rabiFreq12, rabiFreq23, N=1/2):
    # Calculate the gate error due to a servo bump at the specified integrated
    # noise power sg and center frequency fg
    transition = calcs.RydbergTransition()
    Delta = transition.get_OptimalDetuning(rabiFreq1=rabiFreq12,
                                      rabiFreq2=rabiFreq23)
    rabiFreqTotal = rabiFreq12*rabiFreq23/(2*Delta)

    epsilon = 2*sg*(np.pi*fg*rabiFreqTotal)**2
    epsilon *= (1-(-1)**(2*N) * np.cos(4*np.pi**2*N*fg/rabiFreqTotal))
    epsilon /= (rabiFreqTotal**2 - 4*np.pi**2*fg**2)**2

    return epsilon


def intensityGateError(sigma1, sigma2, N=1/2):
    # Calculate the gate error due to relative intensity noise in a two-photon
    # excitation

    epsilon = np.pi**2 * N**2 * (sigma1**2 + sigma2**2) / 4
    return epsilon
