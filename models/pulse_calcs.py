import numpy as np
from numba import njit


@njit
def get_blackman_pulse(t, duration, delay, hold=0):
    """
    Returns the value of a Blackman-Harris window function at time t.

    Parameters
    ----------
    t : float
        The time at which to evaluate the pulse
    duration : float
        The duration of the pulse
    delay : float
        The delay before the pulse starts
    hold : float, optional
        The duration of the flat top of the pulse. Defaults to 0.

    Returns
    -------
    pulse : float
        The value of the pulse at time t
    """
    pulse = 0
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08
    if t < 0:
        pulse = 0
    elif 0 <= t < delay:
        pulse = 0
    elif delay <= t < delay + duration / 2:
        pulse = a0 - a1 * np.cos(
            2 * np.pi * (t - delay) / duration) + a2 * np.cos(
            4 * np.pi * (t - delay) / duration)
    elif delay + duration / 2 <= t < delay + hold + duration / 2:
        pulse = 1
    elif delay + hold + duration / 2 <= t < delay + hold + duration:
        pulse = a0 - a1 * np.cos(
            2 * np.pi * (t - delay - hold) / duration) + a2 * np.cos(
            4 * np.pi * (t - delay - hold) / duration)
    elif delay + hold + duration <= t:
        pulse = 0

    return pulse


@njit
def get_vectorized_blackman_pulse(t, duration, delay, hold=0):
    """
    Compute the values of a Blackman-Harris window function for a vector of time
    points.

    Parameters
    ----------
    t : array_like
        An array of time points at which to evaluate the pulse.
    duration : float
        The duration of the pulse.
    delay : float
        The delay before the pulse starts.
    hold : float, optional
        The duration of the flat top of the pulse. Defaults to 0.

    Returns
    -------
    pulse_pts : ndarray
        An array containing the values of the pulse at each time point in `t`.
    """
    pulse_pts = np.zeros_like(t)

    for j in range(len(pulse_pts)):
        pulse_pts[j] = get_blackman_pulse(t[j], duration=duration,
                                          delay=delay, hold=hold)

    return pulse_pts
