# Plots for rydberg_dynamics.py

import numpy as np
import matplotlib.pyplot as plt
import models.rydberg_dynamics as rydnamics
import matplotlib.colors as colors
import models.pulse_calcs as pulses


def plot_pulse():
	time = np.linspace(0, 1, 10000)
	plt.plot(time, pulses.get_VectorizedBlackmanPulse(time, duration=0.2,
												   delay=0.3,
											   hold=0.))
	plt.ylabel("Amplitude")
	plt.xlabel("Time")
	plt.show()
def plot_state_dynamics():
	runner = rydnamics.ExciteRydberg()

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

	runner = rydnamics.ExciteRydberg()

	coupling_power = 1
	probe_powers = np.linspace(1e-3, 10e-3, 20)
	holds = np.linspace(0, 200e-9, 20)

	Rydberg_final = []
	pi_pulse_duration = []

	for Pp in probe_powers:
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
	Ryd_pop = np.reshape(Ryd_pop, (len(probe_powers),len(holds))).T

	fig, (ax1) = plt.subplots(nrows=1)
	s1 = ax1.imshow(Ryd_pop, vmax=np.max(Ryd_pop), aspect='auto', origin="lower", extent=[np.min(probe_powers),np.max(probe_powers), np.min(holds), np.max(holds)])
	cbar = fig.colorbar(s1, ax=ax1)
	cbar.set_label(r'Rydberg Population')
	ax1.set_xlabel(r'Peak 456nm Power (W)')
	ax1.set_ylabel(r'Hold Time (s)')
	for n in range(1,5,1):
		ax1.plot(probe_powers, n*pi_pulse_duration, color='cyan')

	plt.tight_layout()
	plt.show()

def plot_rho_dynamics():
	runner = rydnamics.ExciteRydberg()

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

if __name__ == '__main__':
	plot_state_dynamics()
	plot_rho_dynamics()