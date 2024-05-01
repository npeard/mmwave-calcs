# Plots for rydberg_dynamics.py

import numpy as np
import matplotlib.pyplot as plt
import models.rydberg_dynamics as rydnamics
import models.pulse_calcs as pulses
from tqdm import tqdm
from matplotlib.colors import LogNorm


def plot_pulse():
    time = np.linspace(0, 1, 10000)
    plt.plot(time, pulses.get_VectorizedBlackmanPulse(time, duration=0.2,
                                                      delay=0.3,
                                                      hold=0.))
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.show()


def plot_state_dynamics():
    runner = rydnamics.UnitaryRydberg()

    G, E, R, pulse, time = runner.probe_pulse_unitary(duration=5e-9,
                                                      delay=10e-9,
                                                      hold=27e-9,
                                                      probe_peak_power=1e-3,
                                                      couple_power=0.1,
                                                      Delta=1.3e10)

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax1.plot(time, G, label="Ground Population")
    ax1.plot(time, E, label="Intermediate Population")
    ax1.plot(time, R, label="RydbergPopulation")
    ax2.plot(time, pulse, label="456nm Pulse", color='cyan')
    ax1.legend()
    ax2.legend()
    ax2.set_ylabel("Power (W)")
    plt.show()


def plot_state_hold_vs_probe_power():
    # At "optimal" detuning, do pi pulse durations match those calculated
    # from theory as we change the 456nm peak power?

    runner = rydnamics.UnitaryRydberg()

    coupling_power = 1
    probe_powers = np.linspace(1e-3, 10e-3, 20)
    holds = np.linspace(0, 200e-9, 20)

    Rydberg_final = []
    pi_pulse_duration = []

    for Pp in tqdm(probe_powers):
        pi_pulse_duration.append(runner.transition.get_PiPulseDuration(Pp=Pp,
                                                                       Pc=coupling_power))
        for hold in holds:
            Ground, Inter, Rydberg, Sweep, _ = runner.probe_pulse_unitary(duration=0e-9,
                                                                          delay=5e-9,
                                                                          hold=hold, probe_peak_power=Pp,
                                                                          couple_power=coupling_power)
            Rydberg_final.append(Rydberg[-1])
    pi_pulse_duration = np.asarray(pi_pulse_duration)
    Ryd_pop = np.asarray(Rydberg_final)
    Ryd_pop = np.reshape(Ryd_pop, (len(probe_powers), len(holds))).T

    fig, (ax1) = plt.subplots(nrows=1)
    s1 = ax1.imshow(
        Ryd_pop,
        vmax=np.max(Ryd_pop),
        aspect='auto',
        origin="lower",
        extent=[
            np.min(probe_powers),
            np.max(probe_powers),
            np.min(holds),
            np.max(holds)])
    cbar = fig.colorbar(s1, ax=ax1)
    cbar.set_label(r'Rydberg Population')
    ax1.set_xlabel(r'Peak 456nm Power (W)')
    ax1.set_ylabel(r'Hold Time (s)')
    for n in range(1, 5, 1):
        ax1.plot(probe_powers, n * pi_pulse_duration, color='cyan')

    plt.tight_layout()
    plt.show()


def plot_state_power_vs_power_fixed_pi():
    runner = rydnamics.UnitaryRydberg()

    coupling_powers = np.linspace(0.1, 10, 20)
    probe_powers = np.linspace(1e-3, 10e-3, 20)
    holds = runner.transition.get_PiPulseDuration(Pp=5e-3, Pc=5)

    Ground_list = []
    Inter_list = []
    Rydberg_final = []
    Sweep_list = []
    pi_pulse_duration = []
    Rabi_ratio = []

    for Pp in tqdm(probe_powers):
        for Pc in coupling_powers:
            Ground, Inter, Rydberg, Sweep, _ = runner.probe_pulse_unitary(duration=0e-9,
                                                                          delay=5e-9,
                                                                          hold=holds, probe_peak_power=Pp,
                                                                          couple_power=Pc)
            # Ground_list.append(Ground)
            # Inter_list.append(Inter)
            Rydberg_final.append(Rydberg[-1])
            Rabi_ratio.append(
                runner.func_Omega23_from_Power(Pc) /
                runner.func_Omega12_from_Power(
                    Pp))
        # Sweep_list.append(Sweep)
    Ryd_pop = np.asarray(Rydberg_final)
    Ryd_pop = np.reshape(Ryd_pop, (len(probe_powers), len(coupling_powers))).T

    Rabi_ratio = np.asarray(Rabi_ratio)
    Rabi_ratio = np.reshape(Rabi_ratio,
                            (len(probe_powers), len(coupling_powers))).T

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    s1 = ax1.imshow(Ryd_pop, vmax=np.max(Ryd_pop), aspect='auto',
                    origin="lower",
                    extent=[np.min(probe_powers), np.max(probe_powers),
                            np.min(coupling_powers), np.max(coupling_powers)])
    cbar = fig.colorbar(s1, ax=ax1)
    cbar.set_label(r'Rydberg Population')
    ax1.set_xlabel(r'Peak 456nm Power (W)')
    ax1.set_ylabel(r'1064nm Power (W)')
    ax1.scatter(5e-3, 5, color='cyan', marker='x', s=100)

    s2 = ax2.imshow(Rabi_ratio, aspect='auto', origin="lower",
                    extent=[np.min(probe_powers), np.max(probe_powers),
                            np.min(coupling_powers), np.max(coupling_powers)])
    cbar = fig.colorbar(s2, ax=ax2)
    cbar.set_label(r'Rabi_2/Rabi_1')
    ax2.set_xlabel(r'Peak 456nm Power (W)')
    ax2.set_ylabel(r'1064nm Power (W)')

    plt.tight_layout()
    plt.show()


def plot_state_couple_power_vs_detune(coupling_powers=None, detunings=None,
                                      probe_peak_power=None, duration=None):
    runner = rydnamics.UnitaryRydberg()

    Ground_list = []
    Inter_list = []
    Rydberg_final = []
    Sweep_list = []
    pi_pulse_duration = []
    optimal_detuning = []

    for Delta in tqdm(detunings):
        for Pc in coupling_powers:
            hold = runner.transition.get_PiPulseDuration(Pp=probe_peak_power,
                                                         Pc=Pc)
            Ground, Inter, Rydberg, Sweep, _ = runner.probe_pulse_unitary(
                duration=duration,
                delay=5e-9, hold=hold,
                probe_peak_power=probe_peak_power,
                couple_power=Pc,
                Delta=Delta)
            Rydberg_final.append(Rydberg[-1])
    Ryd_pop = np.asarray(Rydberg_final)
    Ryd_pop = np.reshape(Ryd_pop, (len(detunings), len(coupling_powers))).T

    for Pc in coupling_powers:
        optimal_detuning.append(runner.transition.get_OptimalDetuning(
            P1=probe_peak_power, P2=Pc))
    optimal_detuning = np.asarray(optimal_detuning)

    fig, (ax1) = plt.subplots(nrows=1)
    s1 = ax1.imshow(Ryd_pop, vmax=np.max(Ryd_pop), aspect='auto',
                    origin="lower",
                    extent=[np.min(detunings), np.max(detunings),
                            np.min(coupling_powers), np.max(coupling_powers)])
    cbar = fig.colorbar(s1, ax=ax1)
    cbar.set_label(r'Rydberg Population')
    ax1.set_xlabel(r'Detuning (2pi x GHz)')
    ax1.set_ylabel(r'1064nm Power (W)')
    ax1.plot(optimal_detuning, coupling_powers, color='cyan')

    plt.tight_layout()
    plt.show()


def plot_rho_dynamics():
    runner = rydnamics.UnitaryRydberg()

    G, E, R, pulse, time = runner.probe_pulse_neumann(duration=5e-9,
                                                      delay=10e-9,
                                                      hold=27e-9,
                                                      probe_peak_power=1e-3,
                                                      couple_power=0.1,
                                                      Delta=1.3e10)

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax1.plot(time, G, label="Ground Population")
    ax1.plot(time, E, label="Intermediate Population")
    ax1.plot(time, R, label="RydbergPopulation")
    ax2.plot(time, pulse, label="456nm Pulse", color='cyan')
    ax1.legend()
    ax2.legend()
    ax2.set_ylabel("Power (W)")
    plt.show()


def plot_lindblad_dynamics():
    runner = rydnamics.LossyRydberg()

    Ground, Inter, Rydberg, Sweep, time, Loss = runner.probe_pulse_lindblad(
        duration=5e-9, delay=10e-9, hold=27e-9, probe_peak_power=1e-3,
        couple_power=0.1,
        Delta=1.3e10)
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax1.plot(time, Ground, label="Ground Population")
    ax1.plot(time, Inter, label="Intermediate Population")
    ax1.plot(time, Rydberg, label="Rydberg Population")
    ax1.plot(time, Loss, label="Loss")
    ax2.plot(time, Sweep, label="456nm Pulse", color='cyan')
    ax1.legend()
    ax2.legend()
    ax2.set_ylabel("Power (W)")
    plt.show()


def plot_lindblad_couple_power_vs_detune(coupling_powers=None, detunings=None,
                                         probe_peak_power=None):
    runner = rydnamics.LossyRydberg()

    Rydberg_final = []
    Inter_final = []
    Loss_final = []
    pi_pulse_duration = []
    optimal_detuning = []

    for Delta in tqdm(detunings):
        for Pc in coupling_powers:
            hold = runner.transition.get_PiPulseDuration(Pp=probe_peak_power,
                                                         Pc=Pc)
            Ground, Inter, Rydberg, Sweep, _, Loss = runner.probe_pulse_lindblad(
                duration=0e-9, delay=5e-9, hold=hold,
                probe_peak_power=probe_peak_power,
                couple_power=Pc,
                Delta=Delta)
            Rydberg_final.append(Rydberg[-1])
            Inter_final.append(Inter[-1])
            Loss_final.append(Loss[-1])
    Ryd_pop = np.asarray(Rydberg_final)
    Ryd_pop = np.reshape(Ryd_pop, (len(detunings), len(coupling_powers))).T

    Inter_pop = np.asarray(Inter_final)
    Inter_pop = np.reshape(Inter_pop, (len(detunings), len(coupling_powers))).T

    Loss_pop = np.asarray(Loss_final)
    Loss_pop = np.reshape(Loss_pop, (len(detunings), len(coupling_powers))).T

    for Pc in coupling_powers:
        optimal_detuning.append(runner.transition.get_OptimalDetuning(
            P1=probe_peak_power,
            P2=Pc))
    optimal_detuning = np.asarray(optimal_detuning)

    fig, ax1 = plt.subplots(nrows=1)
    s1 = ax1.imshow(np.real(Ryd_pop), vmax=np.max(np.real(Ryd_pop)),
                    aspect='auto', origin="lower",
                    extent=[np.min(detunings), np.max(detunings),
                            np.min(coupling_powers), np.max(coupling_powers)])
    cbar = fig.colorbar(s1, ax=ax1)
    cbar.set_label(r'Rydberg Population')
    ax1.set_xlabel(r'Detuning (2pi x GHz)')
    ax1.set_ylabel(r'1064nm Power (W)')
    ax1.plot(optimal_detuning, coupling_powers, color='cyan')
    plt.tight_layout()
    plt.show()

    fig, ax2 = plt.subplots(nrows=1)
    s2 = ax2.imshow(np.real(Inter_pop), norm=LogNorm(vmax=1), aspect='auto',
                    origin="lower",
                    extent=[np.min(detunings), np.max(detunings),
                            np.min(coupling_powers), np.max(coupling_powers)])
    cbar = fig.colorbar(s2, ax=ax2)
    cbar.set_label(r'7p Population')
    ax2.set_xlabel(r'Detuning (2pi x GHz)')
    ax2.set_ylabel(r'1064nm Power (W)')
    ax2.plot(optimal_detuning, coupling_powers, color='cyan')
    plt.tight_layout()
    plt.show()

    fig, ax3 = plt.subplots(nrows=1)
    s3 = ax3.imshow(np.real(Loss_pop), norm=LogNorm(vmax=1), aspect='auto',
                    origin="lower",
                    extent=[np.min(detunings), np.max(detunings),
                            np.min(coupling_powers), np.max(coupling_powers)])
    cbar = fig.colorbar(s3, ax=ax3)
    cbar.set_label(r'Loss Population')
    ax3.set_xlabel(r'Detuning (2pi x GHz)')
    ax3.set_ylabel(r'1064nm Power (W)')
    ax3.plot(optimal_detuning, coupling_powers, color='cyan')
    plt.tight_layout()
    plt.show()
    
def plot_lindblad_fast_probe(coupling_powers=None, detunings=None,
                                         probe_peak_power=None):
    runner = rydnamics.LossyRydberg()

    Rydberg_final = []
    Inter_final = []
    Loss_final = []
    pi_pulse_duration = []

    for Pp in tqdm(probe_peak_power):
        for Pc in coupling_powers:
            hold = runner.transition.get_PiPulseDuration(Pp=Pp,
                                                         Pc=Pc, resonance=True)
            Ground, Inter, Rydberg, Sweep, _, Loss = runner.probe_pulse_lindblad(
                duration=0e-9, delay=5e-9, hold=hold,
                probe_peak_power=probe_peak_power,
                couple_power=Pc,
                Delta=0)
            Rydberg_final.append(Rydberg[-1])
            Inter_final.append(Inter[-1])
            Loss_final.append(Loss[-1])
            pi_pulse_duration.append(hold)
    Ryd_pop = np.asarray(Rydberg_final)
    Ryd_pop = np.reshape(Ryd_pop, (len(detunings), len(coupling_powers))).T

    Inter_pop = np.asarray(Inter_final)
    Inter_pop = np.reshape(Inter_pop, (len(detunings), len(coupling_powers))).T

    Loss_pop = np.asarray(Loss_final)
    Loss_pop = np.reshape(Loss_pop, (len(detunings), len(coupling_powers))).T

    fig, ax1 = plt.subplots(nrows=1)
    s1 = ax1.imshow(np.real(Ryd_pop), vmax=np.max(np.real(Ryd_pop)),
                    aspect='auto', origin="lower",
                    extent=[np.min(probe_peak_power), np.max(probe_peak_power),
                            np.min(coupling_powers), np.max(coupling_powers)])
    cbar = fig.colorbar(s1, ax=ax1)
    cbar.set_label(r'Rydberg Population')
    ax1.set_xlabel(r'456nm Power (W)')
    ax1.set_ylabel(r'1064nm Power (W)')
    plt.tight_layout()
    plt.show()

    fig, ax2 = plt.subplots(nrows=1)
    s2 = ax2.imshow(np.real(Inter_pop), norm=LogNorm(vmax=1), aspect='auto',
                    origin="lower",
                    extent=[np.min(probe_peak_power), np.max(probe_peak_power),
                            np.min(coupling_powers), np.max(coupling_powers)])
    cbar = fig.colorbar(s2, ax=ax2)
    cbar.set_label(r'7p Population')
    ax2.set_xlabel(r'456nm Power (W)')
    ax2.set_ylabel(r'1064nm Power (W)')
    plt.tight_layout()
    plt.show()

    fig, ax3 = plt.subplots(nrows=1)
    s3 = ax3.imshow(np.real(Loss_pop), norm=LogNorm(vmax=1), aspect='auto',
                    origin="lower",
                    extent=[np.min(probe_peak_power), np.max(probe_peak_power),
                            np.min(coupling_powers), np.max(coupling_powers)])
    cbar = fig.colorbar(s3, ax=ax3)
    cbar.set_label(r'Loss Population')
    ax3.set_xlabel(r'456nm Power (W)')
    ax3.set_ylabel(r'1064nm Power (W)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_rho_dynamics()
    plot_lindblad_dynamics()
