import numpy as np
from scipy.constants import c as c_c
from scipy.constants import epsilon_0


def wavelength2freq(wavelength):
    return c_c / wavelength


def wavelength2angularfreq(wavelength):
    return wavelength2freq(wavelength) * 2 * np.pi


def freq2wavelength(frequency):
    return c_c / frequency


def power2field(laserPower, laserWaist):
    # P is power in W
    # A is area in cm^2
    # E is field in V/m
    maxIntensity = 2 * laserPower / (np.pi * laserWaist**2)
    electricField = np.sqrt(2.0 * maxIntensity / (c_c * epsilon_0))
    return electricField
