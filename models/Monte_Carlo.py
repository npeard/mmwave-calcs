import numpy as np
import matplotlib.pyplot as plt
from arc import *
import scipy.constants as const
from scipy.integrate import odeint

def Monte_Carlo_iHO_3d(atom, t, omega_H, omega_L, T, N_ensemble, traping_laser, D2_line):
    
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
        x, vx, y, vy, z, vz = f
        ax, ay, az = iHO_a_3d(x, y, z, omega_H, omega_L)
        dfdt = [vx, ax, vy, ay, vz, az]
        return dfdt

    m = atom.mass
    delta_x = np.sqrt(const.k*T/(m*omega_H**2))
    delta_vx = np.sqrt(const.k*T/m)
    delta_y = np.sqrt(const.k*T/(m*omega_H**2))
    delta_vy = np.sqrt(const.k*T/m)
    delta_z = np.sqrt(const.k*T/(m*omega_L**2))
    delta_vz = np.sqrt(const.k*T/m)
    x_ensemble = np.random.normal(0, delta_x, N_ensemble)
    y_ensemble = np.random.normal(0, delta_y, N_ensemble)
    z_ensemble = np.random.normal(0, delta_z, N_ensemble)
    vx_ensemble = np.random.normal(0, delta_vx, N_ensemble)
    vy_ensemble = np.random.normal(0, delta_vy, N_ensemble)
    vz_ensemble = np.random.normal(0, delta_vz, N_ensemble)


    v = wavelength2frequency(traping_laser)
    v0 = wavelength2frequency(D2_line)
    alpha = abs((v + v0)*(v-v0)/v**2)
    print(f"alpha = {alpha}")

    omega_iHO_H = np.sqrt(alpha)*omega_H
    omega_iHO_L = np.sqrt(alpha)*omega_L

    x_ensemble_t = []
    y_ensemble_t = []
    z_ensemble_t = []
    vx_ensemble_t = []
    vy_ensemble_t = []
    vz_ensemble_t = []

    for x, vx, y, vy, z, vz in zip(x_ensemble, vx_ensemble, y_ensemble, vy_ensemble, z_ensemble, vz_ensemble):
        f0 = [x, vx, y, vy, z, vz]
        sol = odeint(newton_3d, f0, t, args = (omega_iHO_H, omega_iHO_L,))
        x_ensemble_t.append(sol[:, 0])
        vx_ensemble_t.append(sol[:, 1])
        y_ensemble_t.append(sol[:, 2])
        vy_ensemble_t.append(sol[:, 3])
        z_ensemble_t.append(sol[:, 4])
        vz_ensemble_t.append(sol[:, 5])

    x_ensemble_t = np.array(x_ensemble_t)
    y_ensemble_t = np.array(y_ensemble_t)
    z_ensemble_t = np.array(z_ensemble_t)
    vx_ensemble_t = np.array(vx_ensemble_t)
    vy_ensemble_t = np.array(vy_ensemble_t)
    vz_ensemble_t = np.array(vz_ensemble_t)


    return x_ensemble_t, vx_ensemble_t, y_ensemble_t, vy_ensemble_t, z_ensemble_t, vz_ensemble_t 

def Monte_Carlo_free_3d(atom, t, omega_H, omega_L, T, N_ensemble, traping_laser, D2_line):
    
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
        x, vx, y, vy, z, vz = f
        ax, ay, az = iHO_a_3d(x, y, z, omega_H, omega_L)
        dfdt = [vx, ax, vy, ay, vz, az]
        return dfdt

    m = atom.mass
    delta_x = np.sqrt(const.k*T/(m*omega_H**2))
    delta_vx = np.sqrt(const.k*T/m)
    delta_y = np.sqrt(const.k*T/(m*omega_H**2))
    delta_vy = np.sqrt(const.k*T/m)
    delta_z = np.sqrt(const.k*T/(m*omega_L**2))
    delta_vz = np.sqrt(const.k*T/m)
    x_ensemble = np.random.normal(0, delta_x, N_ensemble)
    y_ensemble = np.random.normal(0, delta_y, N_ensemble)
    z_ensemble = np.random.normal(0, delta_z, N_ensemble)
    vx_ensemble = np.random.normal(0, delta_vx, N_ensemble)
    vy_ensemble = np.random.normal(0, delta_vy, N_ensemble)
    vz_ensemble = np.random.normal(0, delta_vz, N_ensemble)


    v = wavelength2frequency(traping_laser)
    v0 = wavelength2frequency(D2_line)
    alpha = abs((v + v0)*(v-v0)/v**2)
    print(f"alpha = {alpha}")

    omega_iHO_H = np.sqrt(alpha)*omega_H
    omega_iHO_L = np.sqrt(alpha)*omega_L

    x_ensemble_t = []
    y_ensemble_t = []
    z_ensemble_t = []
    vx_ensemble_t = []
    vy_ensemble_t = []
    vz_ensemble_t = []

    for x, vx, y, vy, z, vz in zip(x_ensemble, vx_ensemble, y_ensemble, vy_ensemble, z_ensemble, vz_ensemble):
        f0 = [x, vx, y, vy, z, vz]
        sol = odeint(newton_3d, f0, t, args = (0, 0,))
        x_ensemble_t.append(sol[:, 0])
        vx_ensemble_t.append(sol[:, 1])
        y_ensemble_t.append(sol[:, 2])
        vy_ensemble_t.append(sol[:, 3])
        z_ensemble_t.append(sol[:, 4])
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
