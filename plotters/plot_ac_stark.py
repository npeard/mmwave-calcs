import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import models.ac_stark as ac_stark


def plot_Cs_ground_state_tweezer_shift():
    shifter = ac_stark.ACStarkShift()
    wavelengths = np.linspace(780, 1080, 1000) * 1e-9
    powers = np.asarray([20e-3])
    shirley_shift_wavelength_6S = shifter.ac_stark_shift_shirley(wavelengths,
                                                                 powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_6S = shifter.ac_stark_shift_shirley(np.asarray([1070])
                                                            * 1e-9,
                                                            powers).squeeze()

    shifter_6P = ac_stark.ACStarkShift(l=1, j=1.5, mj=1.5)
    powers = np.asarray([20e-3])
    shirley_shift_wavelength_6P = shifter.ac_stark_shift_shirley(wavelengths,
                                                                 powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_6P = shifter.ac_stark_shift_shirley(np.asarray([1070])
                                                            * 1e-9,
                                                            powers).squeeze()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_6S * 1e-6,
             '--',
             label=f"state = {shifter.state}")
    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_6P * 1e-6, '--', label=f"state = {shifter_6P.state}")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylim([-2000, 2000])
    ax1.set_ylabel("AC Stark Shift (MHz)")

    ax1.legend()

    ax2.plot(powers * 1e3, shirley_shift_power_6S * 1e-6, '--',
             label=f"state = {shifter.state}")
    ax2.plot(powers * 1e3, shirley_shift_power_6P * 1e-6, '--', label=f"state = {shifter_6P.state}")
    ax2.set_xlabel("tweezer power (mW)")
    ax2.set_ylim([0, 100])
    ax2.set_ylabel("AC Stark Shift (MHz)")

    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_Cs_7p_tweezer_shift():
    shifter = ac_stark.ACStarkShift()
    wavelengths = np.linspace(850, 1400, 2000) * 1e-9
    powers = np.asarray([20e-3])
    shirley_shift_wavelength_6S = shifter.ac_stark_shift_shirley(wavelengths,
                                                                 powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_6S = shifter.ac_stark_shift_shirley(np.asarray([1070])
                                                            * 1e-9,
                                                            powers).squeeze()

    shifter_7P32 = ac_stark.ACStarkShift(n=7, l=1, j=1.5)
    powers = np.asarray([20e-3])
    shirley_shift_wavelength_7P32 = shifter_7P32.ac_stark_shift_shirley(wavelengths,
                                                                        powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_7P32 = shifter_7P32.ac_stark_shift_shirley(np.asarray([
        1070])
        * 1e-9,
        powers).squeeze()

    shifter_7P12 = ac_stark.ACStarkShift(n=7, l=1, j=0.5)
    powers = np.asarray([20e-3])
    shirley_shift_wavelength_7P12 = shifter_7P12.ac_stark_shift_shirley(
        wavelengths,
        powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_7P12 = shifter_7P12.ac_stark_shift_shirley(np.asarray([
        1070])
        * 1e-9,
        powers).squeeze()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_7P32 * 1e-6,
             '--',
             label=f"state = {shifter_7P32.state}")
    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_7P12 * 1e-6,
             '--',
             label=f"state = {shifter_7P12.state}")
    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_6S * 1e-6,
             '--',
             label=f"state = {shifter.state}")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylim([-2000, 2000])
    ax1.set_ylabel("AC Stark Shift (MHz)")

    ax1.legend()

    ax2.plot(powers * 1e3, shirley_shift_power_7P32 * 1e-6, '--',
             label=f"state = {shifter_7P32.state}")
    ax2.plot(powers * 1e3, shirley_shift_power_7P12 * 1e-6, '--',
             label=f"state = {shifter_7P12.state}")
    ax2.plot(powers * 1e3, shirley_shift_power_6S * 1e-6, '--',
             label=f"state = {shifter.state}")
    ax2.set_xlabel("tweezer power (mW)")
    # ax2.set_ylim([0, 100])
    ax2.set_ylabel("AC Stark Shift (MHz)")

    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_Cs_7p_tweezer_shift()
