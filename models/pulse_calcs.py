import numpy as np
from numba import njit


@njit
def get_blackman_pulse(t, duration, delay, hold=0):
    # if hold is not 0, not a true Blackman pulse
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
    pulse_pts = np.zeros_like(t)

    for j in range(len(pulse_pts)):
        pulse_pts[j] = get_blackman_pulse(t[j], duration=duration,
                                          delay=delay, hold=hold)

    return pulse_pts
