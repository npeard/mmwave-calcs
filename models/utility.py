import numpy as np
from scipy.constants import c as c_c
from scipy.constants import epsilon_0
import os, sys


def wavelength2freq(wavelength):
    """
    Converts a wavelength to a frequency in Hz.

    Parameters
    ----------
    wavelength : float
        The wavelength in meters.

    Returns
    -------
    frequency : float
        The frequency in Hz.
    """
    return c_c / wavelength


def wavelength2angularfreq(wavelength):
    """
    Converts a wavelength to an angular frequency in rad/s.

    Parameters
    ----------
    wavelength : float
        The wavelength in meters.

    Returns
    -------
    angular_frequency : float
        The angular frequency in rad/s.
    """
    return wavelength2freq(wavelength) * 2 * np.pi


def freq2wavelength(frequency):
    """
    Converts a frequency to a wavelength in meters.

    Parameters
    ----------
    frequency : float
        The frequency in Hz.

    Returns
    -------
    wavelength : float
        The wavelength in meters.
    """
    return c_c / frequency


def power2field(laserPower, laserWaist):
    """
    Converts a laser power and waist to an electric field in V/m.

    Parameters
    ----------
    laserPower : float
        The power of the laser, in W.
    laserWaist : float
        The waist of the laser, in meters.

    Returns
    -------
    electricField : float
        The electric field of the laser, in V/m.

    Notes
    -----
    The formula used to calculate the electric field is
    E = sqrt(2 * I / (c * epsilon_0)),
    where I is the intensity in W/m^2, c is the speed of light in m/s,
    and epsilon_0 is the vacuum permittivity in F/m.
    """
    maxIntensity = 2 * laserPower / (np.pi * laserWaist**2)
    electricField = np.sqrt(2.0 * maxIntensity / (c_c * epsilon_0))
    return electricField


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
