# Plots for rydberg_dynamics.py

import numpy as np
import matplotlib.pyplot as plt
import models.rydberg_dynamics as rydnamics
import matplotlib.colors as colors

def plot_state_dynamics():
	runner = rydnamics.ExciteRydberg()

	G, E, R, pulse, time = runner.probe_pulse(duration=5e-9, delay=10e-9,
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