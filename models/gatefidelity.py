# Calculate gate errors based on this paper:
# Jiang, X., Scott, J., Friesen, M. & Saffman, M. Sensitivity of quantum gate
# fidelity to laser phase and intensity noise. Phys. Rev. A 107, 042611 (2023).

import numpy as np
import models.rydberg_calcs as calcs


def white_gate_error(FWHM1, FWHM2, Pp, Pc, N=1 / 2):
    """
    Calculate the gate error for a pi*N gate due to laser phase noise.

    Parameters
    ----------
    FWHM1 : float
        The full width at half maximum of the phase noise power spectrum of the
        first laser, in Hz.
    FWHM2 : float
        The full width at half maximum of the phase noise power spectrum of the
        second laser, in Hz.
    Pp : float
        The power of the probe laser, in Watts.
    Pc : float
        The power of the control laser, in Watts.
    N : float, optional
        The number of pi rotations, defaults to 1/2.

    Returns
    -------
    epsilon : float
        The gate error due to phase noise in the two lasers, in the limit that
        the noise is "white" (i.e. the power spectrum is flat).
    """
    transition = calcs.RydbergTransition()
    rabiFreqTotal = transition.get_total_rabi_angular_freq(Pp, Pc)  # in 2pi*Hz
    h1 = FWHM1 / (2 * np.pi)
    h2 = FWHM2 / (2 * np.pi)

    epsilon = np.pi**3 * (h1 + h2) * N / rabiFreqTotal

    return epsilon


def servo_gate_error(sg, fg, rabiFreq12, rabiFreq23, N=1 / 2):
    """
    Calculate the gate error due to a servo bump at the specified integrated
    noise power sg and center frequency fg.

    Parameters
    ----------
    sg : float
        The integrated noise power of the servo bump, in Hz^2/Hz.
    fg : float
        The center frequency of the servo bump, in Hz.
    rabiFreq12 : float
        The Rabi frequency of the first laser, in 2pi*Hz.
    rabiFreq23 : float
        The Rabi frequency of the second laser, in 2pi*Hz.
    N : float, optional
        The number of pi rotations, defaults to 1/2.

    Returns
    -------
    epsilon : float
        The gate error due to the servo bump, in the limit that the noise is
        confined to a narrow frequency band near the servo bump center
        frequency.
    """
    transition = calcs.RydbergTransition()
    Delta = transition.get_optimal_detuning(rabiFreq1=rabiFreq12,
                                            rabiFreq2=rabiFreq23)
    rabiFreqTotal = rabiFreq12 * rabiFreq23 / (2 * Delta)

    epsilon = 2 * sg * (np.pi * fg * rabiFreqTotal)**2
    epsilon *= (1 - (-1)**(2 * N) * np.cos(4 * np.pi**2 * N * fg / rabiFreqTotal))
    epsilon /= (rabiFreqTotal**2 - 4 * np.pi**2 * fg**2)**2

    return epsilon


def intensity_gate_error(sigma1, sigma2, N=1 / 2):
    """
    Calculate the gate error due to relative intensity noise in a two-photon
    excitation.

    Parameters
    ----------
    sigma1 : float
        The relative intensity variance of the first laser, in Hz^2/Hz.
    sigma2 : float
        The relative intensity variance of the second laser, in Hz^2/Hz.
    N : float, optional
        The number of pi rotations, defaults to 1/2.

    Returns
    -------
    epsilon : float
        The gate error due to relative intensity noise, in the limit that the
        noise is confined to a narrow frequency band near the two-photon Rabi
        frequency.
    """
    epsilon = np.pi**2 * N**2 * (sigma1**2 + sigma2**2) / 4
    return epsilon
