# Plots for gatefidelity.py

import numpy as np
import matplotlib.pyplot as plt
import models.gatefidelity as gf
import models.rydberg_calcs as calcs
import matplotlib.colors as colors


def plot_whiteGateError():
	transition = calcs.RydbergTransition()

	FWHM1 = np.linspace(1e-3, 300e3, 500)  # Hz
	FWHM2 = np.linspace(1e-3, 50e3, 500)  # Hz

	Pp = 10e-3
	Pc = 3

	xv, yv = np.meshgrid(FWHM1, FWHM2)
	GateError = gf.whiteGateError(xv, yv, Pp, Pc, N=1)

	fig, (ax1) = plt.subplots(nrows=1)
	s1 = ax1.imshow(np.asarray(GateError), aspect='auto', origin="lower",
					extent=[np.min(FWHM1), np.max(FWHM1), np.min(FWHM2),
							np.max(FWHM2)], norm=colors.LogNorm())
	# ax1.set_xscale('log')
	# ax1.set_yscale('log')
	cbar = fig.colorbar(s1, ax=ax1)
	cbar.set_label(r'$\varepsilon$')
	ax1.axvline(x=200e3, color='cyan')
	ax1.axhline(y=10e3, color='red')
	ax1.set_xlabel(r'FWHM$_1$ [Hz]')
	ax1.set_ylabel(r'FWHM$_2$ [Hz]')
	fig.suptitle(r'Rabi Frequency $\Omega_0 = 2\pi *$' + str(
		transition.get_totalRabiAngularFreq(Pp, Pc) / (2 * np.pi) * 1e-6) + " "
																		  "MHz")

	plt.tight_layout()
	plt.show()

	print(gf.whiteGateError(10e3, 200e3, Pp, Pc, N=1))
	print(gf.whiteGateError(10e3, 1, Pp, Pc, N=1))
	print(gf.whiteGateError(1, 1, Pp, Pc, N=1))


def plot_servoGateError_Rabi():
	rabi12 = np.linspace(10e3, 1e9, 500)  # Hz
	rabi23 = np.linspace(10e3, 1e9, 500)  # Hz

	xv, yv = np.meshgrid(rabi12, rabi23)
	GateError1MHz = gf.servoGateError(sg=1, fg=1e6, rabiFreq12=xv,
									  rabiFreq23=yv,
								   N=1)
	GateError500kHz = gf.servoGateError(sg=1, fg=0.5e6, rabiFreq12=xv,
									 rabiFreq23=yv,
									 N=1)

	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
	s1 = ax1.imshow(np.asarray(GateError1MHz), aspect='auto', origin="lower",
					extent=[np.min(rabi12), np.max(rabi12), np.min(rabi23),
							np.max(rabi23)], norm=colors.LogNorm())
	cbar = fig.colorbar(s1, ax=ax1)
	cbar.set_label(r'$\varepsilon$')
	ax1.axvline(x=1e6, color='cyan')
	ax1.axhline(y=1e6, color='red')
	ax1.set_xlabel(r'$\Omega_1$ [$2\pi$ Hz]')
	ax1.set_ylabel(r'$\Omega_2$ [$2\pi$ Hz]')
	ax1.set_title(r'$f_g = 1$ MHz')

	s2 = ax2.imshow(np.asarray(GateError500kHz), aspect='auto', origin="lower",
					extent=[np.min(rabi12), np.max(rabi12), np.min(rabi23),
							np.max(rabi23)], norm=colors.LogNorm())
	cbar = fig.colorbar(s2, ax=ax2)
	cbar.set_label(r'$\varepsilon$')
	ax2.axvline(x=500e3, color='cyan')
	ax2.axhline(y=500e3, color='red')
	ax2.set_xlabel(r'$\Omega_1$ [$2\pi$ Hz]')
	ax2.set_ylabel(r'$\Omega_2$ [$2\pi$ Hz]')
	ax2.set_title(r'$f_g = 500$ kHz')

	plt.tight_layout()
	plt.show()


def plot_servoGateError_Power():
	laserPower1 = np.linspace(1e-9, 10e-3, 5000)  # W
	laserPower2 = np.linspace(1e-9, 1, 5000)  # W

	transition = calcs.RydbergTransition()

	rabi12 = transition.get_E_RabiAngularFreq(laserPower1)
	rabi23 = transition.get_R_RabiAngularFreq(laserPower2)

	xv, yv = np.meshgrid(rabi12, rabi23)
	GateError = gf.servoGateError(sg=1, fg=1e6, rabiFreq12=xv, rabiFreq23=yv,
							   N=1)

	fig, (ax1) = plt.subplots(nrows=1)
	s1 = ax1.imshow(np.asarray(GateError), aspect='auto', origin="lower",
					extent=[np.min(laserPower1), np.max(laserPower1),
							np.min(laserPower2),
							np.max(laserPower2)], norm=colors.LogNorm())
	cbar = fig.colorbar(s1, ax=ax1)
	cbar.set_label(r'$\varepsilon$')
	# ax1.axvline(x=1e6, color='cyan')
	# ax1.axhline(y=1e6, color='red')
	ax1.set_xlabel(r'$P_1$ [W]')
	ax1.set_ylabel(r'$P_2$ [W]')

	plt.tight_layout()
	plt.show()


def plot_intensityGateError():
	sigma12 = np.linspace(1e-8, 1e-3, 500)  # relative intensity variance
	sigma23 = np.linspace(1e-8, 1e-3, 500)  # relative intensity variance

	xv, yv = np.meshgrid(sigma12, sigma23)
	GateError = gf.intensityGateError(sigma1=xv, sigma2=yv, N=1)

	fig, (ax1) = plt.subplots(nrows=1)
	s1 = ax1.imshow(np.asarray(GateError), aspect='auto', origin="lower",
					extent=[np.min(sigma12), np.max(sigma12), np.min(sigma23),
							np.max(sigma23)], norm=colors.LogNorm())
	cbar = fig.colorbar(s1, ax=ax1)
	cbar.set_label(r'$\varepsilon$')
	# ax1.axvline(x=1e6, color='cyan')
	# ax1.axhline(y=1e6, color='red')
	ax1.set_xlabel(r'$\sigma_1$ [RIN]')
	ax1.set_ylabel(r'$\sigma_2$ [RIN]')

	plt.tight_layout()
	plt.show()