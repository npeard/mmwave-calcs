import numpy as np
import matplotlib.pyplot as plt
from arc import *
import scipy.constants as const
from scipy.integrate import odeint

def Monte_Carlo_iHO_3d(atom, t, omega_H, omega_L, T, N_ensemble, traping_laser, D2_line, do_iHO = False):
    """
    Perform a Monte Carlo simulation of a 3D harmonic oscillator with an inhomogeneous
    potential. The simulation is performed using the odeint function from the scipy.integrate
    module. The function takes in the following parameters:
    
    Parameters:
    ----------
    atom : object
        An instance of the Atom class representing the atom being simulated.
    t : array-like
        An array of time values at which to evaluate the simulation.
    omega_H : float
        The frequency of the horizontal oscillation.
    omega_L : float
        The frequency of the vertical oscillation.
    T : float
        The temperature of the system.
    N_ensemble : int
        The number of ensemble members to generate.
    traping_laser : float
        The wavelength of the trapping laser.
    D2_line : float
        The wavelength of the D2 line.
    do_iHO : bool, optional
        Flag indicating whether to include the inverter harmonics potential (default: False).
    
    Returns:
    -------
    x_ensemble_t : ndarray
        An array of shape (N_ensemble, len(t)) containing the x coordinates of the ensemble
        members at each time step.
    vx_ensemble_t : ndarray
        An array of shape (N_ensemble, len(t)) containing the x velocities of the ensemble
        members at each time step.
    y_ensemble_t : ndarray
        An array of shape (N_ensemble, len(t)) containing the y coordinates of the ensemble
        members at each time step.
    vy_ensemble_t : ndarray
        An array of shape (N_ensemble, len(t)) containing the y velocities of the ensemble
        members at each time step.
    z_ensemble_t : ndarray
        An array of shape (N_ensemble, len(t)) containing the z coordinates of the ensemble
        members at each time step.
    vz_ensemble_t : ndarray
        An array of shape (N_ensemble, len(t)) containing the z velocities of the ensemble
        members at each time step.
    """
    def wavelength2frequency (wavelength):
        from scipy.constants import c as c_c
        return c_c/(wavelength)
    
    def iHO_a_3d (x, y, z, omega_H, omega_L):
        ax = (omega_H**2)*x
        ay = (omega_H**2)*y
        g = 9.8 #m/s^2 gravity
        az = (omega_L**2)*z - g
        return ax, ay, az
    
    def newton_3d(f, t, omega_H, omega_L):
        x, y, z, vx, vy, vz = f
        ax, ay, az = iHO_a_3d(x, y, z, omega_H, omega_L)
        dfdt = [vx, vy, vz, ax, ay, az]
        return dfdt

    m = atom.mass
    delta_r = np.sqrt(const.k*T/(m*np.array([omega_H, omega_H, omega_L])**2))
    delta_v = np.sqrt(const.k*T/m)*np.ones(3)
    x_ensemble, y_ensemble, z_ensemble =  (delta_r.T * np.random.normal(0, 1, size = (N_ensemble, 3)) ).T
    vx_ensemble, vy_ensemble, vz_ensemble= (delta_v.T * np.random.normal(0, 1, size = (N_ensemble, 3))).T


    f = wavelength2frequency(traping_laser)
    f0 = wavelength2frequency(D2_line)
    alpha = abs((f + f0)*(f-f0)/f**2) # see paper https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.053803
    print(f"alpha = {alpha}")

    omega_iHO_H = np.sqrt(alpha)*omega_H
    omega_iHO_L = np.sqrt(alpha)*omega_L

    x_ensemble_t = []
    y_ensemble_t = []
    z_ensemble_t = []
    vx_ensemble_t = []
    vy_ensemble_t = []
    vz_ensemble_t = []

    for f0 in zip(x_ensemble, y_ensemble, z_ensemble, vx_ensemble, vy_ensemble, vz_ensemble):
        if do_iHO:
            sol = odeint(newton_3d, f0, t, args = (omega_iHO_H, omega_iHO_L,))
        else:
            sol = odeint(newton_3d, f0, t, args = (0, 0,))
        x_ensemble_t.append(sol[:, 0])
        y_ensemble_t.append(sol[:, 1])
        z_ensemble_t.append(sol[:, 2])
        vx_ensemble_t.append(sol[:, 3])
        vy_ensemble_t.append(sol[:, 4])
        vz_ensemble_t.append(sol[:, 5])

    x_ensemble_t = np.array(x_ensemble_t)
    y_ensemble_t = np.array(y_ensemble_t)
    z_ensemble_t = np.array(z_ensemble_t)
    vx_ensemble_t = np.array(vx_ensemble_t)
    vy_ensemble_t = np.array(vy_ensemble_t)
    vz_ensemble_t = np.array(vz_ensemble_t)


    return x_ensemble_t, vx_ensemble_t, y_ensemble_t, vy_ensemble_t, z_ensemble_t, vz_ensemble_t 



def recature_rate_3d(trap_temp, x_ensemble_t, vx_ensemble_t, y_ensemble, vy_ensemble, z_ensemble, vz_ensemble, t, m, omega_H, omega_L):
    effective_trap_depth = const.k * trap_temp

    print(f'effective_trap_depth = {effective_trap_depth}')
    def energy_3d (x, vx, y, vy, z, vz, m, omega_H, omega_L):
        E = 0.5*m*(vx**2 + vy**2 + vz**2) + 0.5*m*(omega_H**2*(x**2+y**2)+ omega_L**2*(z**2))
        return E
    
    energy_ind = np.zeros((len(t),x_ensemble_t.shape[0]))
    # print(x_ensemble_t.shape[0])
    for i in np.arange(len(t)):
        for j in np.arange(x_ensemble_t.shape[0]):
            energy_ind[i, j] = energy_3d(x_ensemble_t[j, i], vx_ensemble_t[j, i], y_ensemble[j, i], vy_ensemble[j, i], z_ensemble[j, i], vz_ensemble[j, i], m, omega_H, omega_L)

    # print(energy_ind)
    recapture_number = np.zeros(len(t))
    for i in np.arange(len(t)):
        recapture_number[i] = np.sum(energy_ind[i, :] <= effective_trap_depth)

    recapture_rate = recapture_number/x_ensemble_t.shape[0]

    return recapture_rate
