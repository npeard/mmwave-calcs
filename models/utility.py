import numpy as np
from scipy.constants import c as C_c
from scipy.constants import epsilon_0


def wavelength2freq(wavelength):
    return 299792458 / wavelength


def freq2wavelength(frequency):
    return 299792458 / frequency


def power2field(laserPower, laserWaist):
    # P is power in W
    # A is area in cm^2
    # E is field in V/m
    maxIntensity = 2 * laserPower / (np.pi * laserWaist**2)
    electricField = np.sqrt(2.0 * maxIntensity / (C_c * epsilon_0))
    return electricField
