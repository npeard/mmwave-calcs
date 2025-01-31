import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import models.ac_stark as ac_stark


def plot_Cs_ground_state_tweezer_shift():
    shifter = ac_stark.ACStarkShift()#laserWaist=1e-6, n=6, l=0, j=0.5,
    # mj=0.5, q=0)
    tweezer_wavelength = np.asarray([1069.79]) * 1e-9
    wavelengths = np.linspace(450, 460, 3000) * 1e-9
    powers = np.asarray([20e-3])
    shirley_shift_wavelength_6S = shifter.ac_stark_shift_shirley(wavelengths,
                                                                 powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_6S = shifter.ac_stark_shift_shirley(tweezer_wavelength,
                                                            powers).squeeze()

    shifter_6P = ac_stark.ACStarkShift(l=1, j=1.5, mj=1.5)
    powers = np.asarray([20e-3])
    shirley_shift_wavelength_6P = shifter.ac_stark_shift_shirley(wavelengths,
                                                                 powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_6P = shifter.ac_stark_shift_shirley(tweezer_wavelength,
                                                            powers).squeeze()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_6S * 1e-6,
             '--',
             label=f"state = {shifter.target_state}")
    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_6P * 1e-6, '--', label=f"state = {shifter_6P.target_state}")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylim([-2000, 2000])
    ax1.set_ylabel("AC Stark Shift (MHz)")

    ax1.legend()
    ax1.grid()

    ax2.plot(powers * 1e3, shirley_shift_power_6S * 1e-6, '--',
             label=f"state = {shifter.target_state}")
    ax2.plot(powers * 1e3, shirley_shift_power_6P * 1e-6, '--',
             label=f"state = {shifter_6P.target_state}")
    ax2.set_xlabel("tweezer power (mW)")
    # ax2.set_ylim([0, 100])
    ax2.set_ylabel("AC Stark Shift (MHz)")
    ax2.set_title(
        "tweezer wavelength = " + str(tweezer_wavelength[0] * 1e9) + " nm")

    ax2.legend()
    ax2.grid()

    plt.suptitle(
        "Cs GS AC Stark Shifts, waist = " + str(shifter.laserWaist * 1e6) + " "
                                                                           "um")
    plt.tight_layout()
    plt.show()


def plot_Cs_7p_tweezer_shift():
    tweezer_wavelength = np.asarray([1069.79]) * 1e-9
    tweezer_power = np.asarray([20e-3])

    shifter = ac_stark.ACStarkShift()
    wavelengths = np.linspace(850, 1400, 2000) * 1e-9
    powers = tweezer_power
    shirley_shift_wavelength_6S = shifter.ac_stark_shift_shirley(wavelengths,
                                                                 powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_6S = shifter.ac_stark_shift_shirley(tweezer_wavelength,
                                                            powers).squeeze()

    shifter_7P32 = ac_stark.ACStarkShift(n=7, l=1, j=1.5)
    powers = tweezer_power
    shirley_shift_wavelength_7P32 = shifter_7P32.ac_stark_shift_shirley(
        wavelengths, powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_7P32 = shifter_7P32.ac_stark_shift_shirley(tweezer_wavelength,
                                                                   powers).squeeze()

    shifter_7P12 = ac_stark.ACStarkShift(n=7, l=1, j=0.5)
    powers = tweezer_power
    shirley_shift_wavelength_7P12 = shifter_7P12.ac_stark_shift_shirley(
        wavelengths,
        powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_7P12 = shifter_7P12.ac_stark_shift_shirley(tweezer_wavelength,
                                                                   powers).squeeze()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_7P32 * 1e-6,
             '--',
             label=f"state = {shifter_7P32.target_state}")
    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_7P12 * 1e-6,
             '--',
             label=f"state = {shifter_7P12.target_state}")
    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_6S * 1e-6,
             '--',
             label=f"state = {shifter.target_state}")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylim([-200, 200])
    ax1.set_ylabel("AC Stark Shift (MHz)")
    ax1.set_title("tweezer power = " + str(tweezer_power[0] * 1e3) + " mW")

    ax1.legend()
    ax1.grid()

    ax2.plot(powers * 1e3, shirley_shift_power_7P32 * 1e-6, '--',
             label=f"state = {shifter_7P32.target_state}")
    ax2.plot(powers * 1e3, shirley_shift_power_7P12 * 1e-6, '--',
             label=f"state = {shifter_7P12.target_state}")
    ax2.plot(powers * 1e3, shirley_shift_power_6S * 1e-6, '--',
             label=f"state = {shifter.target_state}")
    ax2.set_xlabel("tweezer power (mW)")
    # ax2.set_ylim([0, 100])
    ax2.set_ylabel("AC Stark Shift (MHz)")
    ax2.set_title("tweezer wavelength = " + str(tweezer_wavelength[0] * 1e9) + " nm")

    ax2.legend()
    ax2.grid()

    plt.suptitle("7P AC Stark Shifts, waist = " + str(shifter.laserWaist * 1e6) + " um")
    plt.tight_layout()
    plt.show()


def plot_Cs_Rydberg_local_address_shift(n=40):
    addr_wavelength = np.asarray([1064]) * 1e-9
    addr_power = np.asarray([2e-3])
    wavelengths = np.linspace(850, 1100, 2000) * 1e-9

    # Rydberg states
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    state_list = [{"n": n, "l": 0, "j": 0.5, "mj": 0.5},
                  {"n": n, "l": 1, "j": 0.5, "mj": 0.5},
                  {"n": n, "l": 1, "j": 1.5, "mj": 0.5},
                  {"n": n, "l": 1, "j": 1.5, "mj": 1.5},
                  {"n": n, "l": 2, "j": 1.5, "mj": 1.5},
                  {"n": n, "l": 2, "j": 2.5, "mj": 1.5},
                  {"n": n, "l": 2, "j": 2.5, "mj": 2.5}]
    for state in state_list:
        shifter = ac_stark.ACStarkShift(**state)
        print(shifter.target_state)
        powers = addr_power
        shirley_shift_wavelength = shifter.ac_stark_shift_shirley(wavelengths,
                                                                  powers).squeeze()

        powers = np.linspace(0, 20e-3, 1000)
        shirley_shift_power = shifter.ac_stark_shift_shirley(addr_wavelength,
                                                             powers).squeeze()

        ax1.plot(wavelengths * 1e9, shirley_shift_wavelength * 1e-6, '--',
                 label=f"state = {shifter.target_state}")
        ax2.plot(powers * 1e3, shirley_shift_power * 1e-6, '--',
                 label=f"state = {shifter.target_state}")

    # Ground state
    shifter = ac_stark.ACStarkShift()
    powers = addr_power
    shirley_shift_wavelength_6S = shifter.ac_stark_shift_shirley(wavelengths,
                                                                 powers).squeeze()

    powers = np.linspace(0, 20e-3, 1000)
    shirley_shift_power_6S = shifter.ac_stark_shift_shirley(addr_wavelength,
                                                            powers).squeeze()


    ax1.plot(wavelengths * 1e9, shirley_shift_wavelength_6S * 1e-6, '--',
            label=f"state = {shifter.target_state}")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylim([-100, 100])
    ax1.set_ylabel("AC Stark Shift (MHz)")
    ax1.set_title("tweezer power = " + str(addr_power[0] * 1e3) + " mW")

    ax1.legend()
    ax1.grid()

    ax2.plot(powers * 1e3, shirley_shift_power_6S * 1e-6, '--',
            label=f"state = {shifter.target_state}")
    ax2.set_xlabel("tweezer power (mW)")
    # ax2.set_ylim([0, 100])
    ax2.set_ylabel("AC Stark Shift (MHz)")
    ax2.set_title("tweezer wavelength = " + str(addr_wavelength[0] * 1e9) + " nm")

    ax2.legend()
    ax2.grid()

    plt.suptitle("Rydberg AC Stark Shifts, waist = " + str(shifter.laserWaist *
                                                     1e6) + " um")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #plot_Cs_ground_state_tweezer_shift()
    plot_Cs_Rydberg_local_address_shift(n=41)
