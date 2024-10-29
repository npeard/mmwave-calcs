import numpy as np
from scipy.constants import c as C_c
from scipy.constants import epsilon_0
import scipy.constants as const


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

def p2i_gaussian(P,w0):
    """
    convert power to intensity in a gaussian beam
    
    Parameters:
        P: Power of laser (W)
        w0: waist of laser (m)
    """
    return 2 * P / (np.pi * w0**2) # 2P/(pi*w0^2)

def w2omega(wavelength):
    """
        Convert wavelength to angular frequency.

        This function calculates the angular frequency of light given its wavelength
        using the formula: ω = 2πc/λ, where c is the speed of light in vacuum.

        Args:
            wavelength (float): Wavelength of the light in meters (m).

        Returns:
            float: Angular frequency in radians per second (rad/s).

    """
    return 2*np.pi*const.c/wavelength # 2*pi*c/w 

def E2T(U0, debug = False):
    """
    Convert energy to temperature.

    Args:
        energy (float): Energy in Joules (J).
        debug (bool, optional): If True, print debug information. Defaults to False.

    Returns:
        float: Temperature in Kelvin (K).

    Raises:
        ValueError: If the input energy is negative.

    Examples:
        >>> energy_to_temperature(1.38e-23)
        1.0
        >>> energy_to_temperature(2.76e-23, debug=True)
        T = 2.000 mK
        2.0
    """
    if np.isscalar(U0) and U0 < 0:
        raise ValueError("Energy must be non-negative.")
    elif isinstance(U0, np.ndarray) and np.any(U0 < 0):
        raise ValueError("All energy values must be non-negative.")
    
    T = U0/const.k

    if debug == True:
        print(f"T = {T*1e6} μK")

    return T

def E2f(U0, debug = False):
    """
    Convert energy to frequency.

    Args:
        energy (float): Energy in Joules (J).
        debug (bool, optional): If True, print debug information. Defaults to False.

    Returns:
        float: Frequency in Hertz (Hz).

    Raises:
        ValueError: 

    Examples:
        >>> energy_to_frequency(1.38e-23)
        1.0
        >>> energy_to_frequency(2.76e-23, debug=True)
        f = 2.000 GHz
        2.0
    """
    f = U0/const.h

    if debug == True:
        print(f"f = {f/1e6} MHz")

    return f

def Rayleigh_range(w0, lamda, debug = False):
    """
    return the dipole trap potential using Eq 9.46 from Atomic physics (Foot)
    
    Parameters:
        w0: waist
        lamda: wavelength
    """

    zR = np.pi*w0**2/lamda
    if debug == True:
        print(f"zR = {zR * 1e3} mm")
    
    return zR

def trap_frequency(U0, w0, zR, m, debug = False):
    """
    return the trap frequency
    
    Parameters:
        U0: trap potential
        waist: minimum waist
        zR: Rayleigh_range
        m: mass of atom
    """
    if np.isscalar(U0) and U0 < 0:
        raise ValueError("Energy must be non-negative.")
    elif isinstance(U0, np.ndarray) and np.any(U0 < 0):
        raise ValueError("All energy values must be non-negative.")
    
    omega_H = np.sqrt(4*U0/(m*w0**2))
    omega_L = np.sqrt(2*U0/(m*zR**2))

    if debug == True:
        print(f"radial trap frequency 2π x {omega_H/1e3/(2*np.pi)} kHz, axial trap frequency 2π x {omega_L/1e3/(2*np.pi)} kHz")

    return omega_H, omega_L
