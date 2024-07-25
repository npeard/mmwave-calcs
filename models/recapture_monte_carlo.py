import numpy as np
from models.utility import wavelength2freq
import scipy.constants as const
from scipy.integrate import odeint
from arc import Caesium
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence



def iHO_a_3d(x, y, z, omega_H, omega_L):
    ax = (omega_H**2) * x
    ay = (omega_H**2) * y
    g = 9.8  # m/s^2 gravity
    az = (omega_L**2) * z - g
    return ax, ay, az


def newton_3d(f, t, omega_H, omega_L):
    x, y, z, vx, vy, vz = f
    ax, ay, az = iHO_a_3d(x, y, z, omega_H, omega_L)
    dfdt = [vx, vy, vz, ax, ay, az]
    return dfdt


def monte_carlo_3d(t, omega_H, omega_L, T, N_ensemble=2000,
                   trapping_laser=1070e-9, D2_line=852e-9, mass=Caesium().mass, do_iHO=False):

    delta_r = np.sqrt(const.k * T / (mass * np.array([omega_H, omega_H, omega_L])**2))
    delta_v = np.sqrt(const.k * T / mass) * np.ones(3)
    x_ensemble, y_ensemble, z_ensemble = (delta_r.T * np.random.normal(0, 1, size=(N_ensemble, 3))).T
    vx_ensemble, vy_ensemble, vz_ensemble = (delta_v.T * np.random.normal(0, 1, size=(N_ensemble, 3))).T

    f = wavelength2freq(trapping_laser)
    f0 = wavelength2freq(D2_line)
    # see https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.053803
    # for definition of alpha
    alpha = abs((f + f0) * (f - f0) / f**2)

    omega_iHO_H = np.sqrt(alpha) * omega_H
    omega_iHO_L = np.sqrt(alpha) * omega_L

    x_ensemble_t = []
    y_ensemble_t = []
    z_ensemble_t = []
    vx_ensemble_t = []
    vy_ensemble_t = []
    vz_ensemble_t = []

    for f0 in zip(x_ensemble, y_ensemble, z_ensemble, vx_ensemble, vy_ensemble, vz_ensemble):
        if do_iHO:
            sol = odeint(newton_3d, f0, t, args=(omega_iHO_H, omega_iHO_L,))
        else:
            sol = odeint(newton_3d, f0, t, args=(0, 0,))
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


def energy_3d(x, vx, y, vy, z, vz, mass, omega_H, omega_L):
    E = 0.5 * mass * (vx**2 + vy**2 + vz**2) + 0.5 * mass * (omega_H**2 * (
        x**2 + y**2) + omega_L**2 * (z**2))
    return E


def recapture_rate_3d(t, omega_H, omega_L, trap_temp, T_ensemble,
                      mass=Caesium().mass, do_iHO=False):

    effective_trap_depth = const.k * trap_temp

    (x_ensemble_t,
     vx_ensemble_t,
     y_ensemble,
     vy_ensemble,
     z_ensemble,
     vz_ensemble) = monte_carlo_3d(t, omega_H, omega_L, T=T_ensemble, do_iHO=do_iHO)

    energy_ind = np.zeros((len(t), x_ensemble_t.shape[0]))

    for i in np.arange(len(t)):
        for j in np.arange(x_ensemble_t.shape[0]):
            energy_ind[i, j] = energy_3d(x_ensemble_t[j, i], vx_ensemble_t[j, i], y_ensemble[j, i],
                                         vy_ensemble[j, i], z_ensemble[j, i],
                                         vz_ensemble[j, i], mass, omega_H,
                                         omega_L)

    recapture_number = np.zeros(len(t))
    for i in np.arange(len(t)):
        recapture_number[i] = np.sum(energy_ind[i, :] <= effective_trap_depth)

    recapture_rate = recapture_number / x_ensemble_t.shape[0]

    return recapture_rate


def fit_gs_tweezer_recapture(t, recapture_rate):
    from lmfit import Model, Parameters

    model = Model(recapture_rate_3d, independent_vars=['t'])
    params = Parameters()
    params.add('omega_H', value=0.8e6, min=0, max=2e6)
    params.add('omega_L', value=0.18e6, min=0, max=2e6)
    params.add('trap_temp', value=2.63e-3, min=0, max=5e-3)  # 2.63e-3
    params.add('T_ensemble', value=20e-6, min=0, max=1e-3)  # 20e-6
    params.add('mass', value=Caesium().mass)
    params['mass'].vary = False
    params['omega_H'].vary = False
    params['omega_L'].vary = False
    params['trap_temp'].vary = False

    result = model.fit(recapture_rate, params, t=t, method='leastsq')

    print(result.fit_report())
    result.plot()
    plt.show()
    return result.best_fit


def plot_recapture_rate():

    omega_H = 0.8e6  # Hz
    omega_L = 0.19e6  # Hz
    T_ensemble = 20e-6  # 168e-6 #K

    trap_temp = 2.63e-3  # K
    t = np.linspace(0, 100, 1000) * 1e-6

    recapture_rate_iHO = recapture_rate_3d(t, omega_H, omega_L, trap_temp, T_ensemble, do_iHO=True)
    recapture_rate_free = recapture_rate_3d(t, omega_H, omega_L, trap_temp, T_ensemble, do_iHO=False)

    plt.plot(t * 1e6, recapture_rate_iHO, label='iHO')
    plt.plot(t * 1e6, recapture_rate_free, label='free')
    plt.xlabel('t (us)')
    plt.ylabel('recapture number')
    plt.legend(loc='best')
    plt.title(f'Trap depth = {trap_temp * 1e3} mK')
    plt.grid()
    plt.show()


def plot_loss_landscape():
    time = np.asarray([
        0.00000000e+00, 1.00100100e-06, 2.00200200e-06, 3.00300300e-06,
        4.00400400e-06, 5.00500501e-06, 6.00600601e-06, 7.00700701e-06,
        8.00800801e-06, 9.00900901e-06, 1.00100100e-05, 1.10110110e-05,
        1.20120120e-05, 1.30130130e-05, 1.40140140e-05, 1.50150150e-05,
        1.60160160e-05, 1.70170170e-05, 1.80180180e-05, 1.90190190e-05,
        2.00200200e-05, 2.10210210e-05, 2.20220220e-05, 2.30230230e-05,
        2.40240240e-05, 2.50250250e-05, 2.60260260e-05, 2.70270270e-05,
        2.80280280e-05, 2.90290290e-05, 3.00300300e-05, 3.10310310e-05,
        3.20320320e-05, 3.30330330e-05, 3.40340340e-05, 3.50350350e-05,
        3.60360360e-05, 3.70370370e-05, 3.80380380e-05, 3.90390390e-05,
        4.00400400e-05, 4.10410410e-05, 4.20420420e-05, 4.30430430e-05,
        4.40440440e-05, 4.50450450e-05, 4.60460460e-05, 4.70470470e-05,
        4.80480480e-05, 4.90490490e-05, 5.00500501e-05, 5.10510511e-05,
        5.20520521e-05, 5.30530531e-05, 5.40540541e-05, 5.50550551e-05,
        5.60560561e-05, 5.70570571e-05, 5.80580581e-05, 5.90590591e-05,
        6.00600601e-05, 6.10610611e-05, 6.20620621e-05, 6.30630631e-05,
        6.40640641e-05, 6.50650651e-05, 6.60660661e-05, 6.70670671e-05,
        6.80680681e-05, 6.90690691e-05, 7.00700701e-05, 7.10710711e-05,
        7.20720721e-05, 7.30730731e-05, 7.40740741e-05, 7.50750751e-05,
        7.60760761e-05, 7.70770771e-05, 7.80780781e-05, 7.90790791e-05,
        8.00800801e-05, 8.10810811e-05, 8.20820821e-05, 8.30830831e-05,
        8.40840841e-05, 8.50850851e-05, 8.60860861e-05, 8.70870871e-05,
        8.80880881e-05, 8.90890891e-05, 9.00900901e-05, 9.10910911e-05,
        9.20920921e-05, 9.30930931e-05, 9.40941e-05, 9.50950951e-05,
        9.60960961e-05, 9.70970971e-05, 9.80980981e-05, 9.90990991e-05])

    recapture = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.997, 0.982, 0.957,
                            0.912, 0.87, 0.826,
                            0.768, 0.702, 0.656, 0.601, 0.559, 0.514, 0.47, 0.438, 0.404, 0.375, 0.348, 0.323,
                            0.302, 0.277, 0.256, 0.241, 0.219, 0.2, 0.188, 0.175, 0.16, 0.15, 0.138, 0.127,
                            0.122, 0.114, 0.107, 0.098, 0.095, 0.091, 0.082, 0.077, 0.072, 0.066, 0.062, 0.057,
                            0.056, 0.055, 0.054, 0.052, 0.051, 0.049, 0.044, 0.043, 0.042, 0.041, 0.039, 0.036,
                            0.035, 0.035, 0.033, 0.031, 0.029, 0.028, 0.027, 0.027, 0.025, 0.025, 0.023, 0.022,
                            0.02, 0.02, 0.02, 0.02, 0.019, 0.018, 0.017, 0.017, 0.017, 0.016, 0.014, 0.014,
                            0.014, 0.014, 0.014, 0.013, 0.013, 0.013, 0.012, 0.011, 0.011, 0.011, 0.011, 0.011,
                            0.011, 0.01, 0.009, 0.009])

    mse = []
    params = np.linspace(0.6e6, 1e6, 100)

    for p in params:
        rates = []
        rates = recapture_rate_3d(time, omega_H=p, omega_L=0.18e6,
                                     trap_temp=2.63e-3, T_ensemble=20e-6,
                                     do_iHO=False)
        rates = np.asarray(rates)
        mse.append(np.mean((rates - recapture)**2))

    mse = np.asarray(mse)

    plt.plot(params, mse)
    plt.show()


def tweezer_recapture_gaussian_process_regress(recapture_time, recapture_rate):
    space = [Real(0.79e6, 0.8e6, name="omega_H"),
             Real(0.17e6, 0.18e6, name="omega_L"),
             Real(1e-6, 10e-3, name="trap_temp"),
             Real(1e-6, 1e-3, name="T_ensemble")]
    
    @use_named_args(space)
    def objective_func(omega_H, omega_L, trap_temp, T_ensemble):
        predicted_rate = recapture_rate_3d(recapture_time, omega_H, omega_L,
                                           trap_temp, T_ensemble, do_iHO=False)
        return np.mean((recapture_rate - predicted_rate)**2)

    result = gp_minimize(objective_func, space, n_calls=100,
                         acq_func="gp_hedge",
                         n_random_starts=10, random_state=0)

    print(result.x)
    print(result.fun)
    
    plot_convergence(result, yscale="log")
    plt.show()
    
    best_fit = recapture_rate_3d(recapture_time, result.x[0], result.x[1],
                                  result.x[2], result.x[3], do_iHO=False)
    
    plt.scatter(recapture_time * 1e6, recapture_rate, label='data')
    plt.plot(recapture_time * 1e6, best_fit, label='GPR fit')
    plt.xlabel('t (us)')
    plt.ylabel('recapture number')
    plt.legend(loc='best')
    # plt.title(f'Trap depth = {trap_temp * 1e3} mK')
    plt.grid()
    plt.show()

def run_fit():
    time = np.asarray([
        0.00000000e+00, 1.00100100e-06, 2.00200200e-06, 3.00300300e-06,
        4.00400400e-06, 5.00500501e-06, 6.00600601e-06, 7.00700701e-06,
        8.00800801e-06, 9.00900901e-06, 1.00100100e-05, 1.10110110e-05,
        1.20120120e-05, 1.30130130e-05, 1.40140140e-05, 1.50150150e-05,
        1.60160160e-05, 1.70170170e-05, 1.80180180e-05, 1.90190190e-05,
        2.00200200e-05, 2.10210210e-05, 2.20220220e-05, 2.30230230e-05,
        2.40240240e-05, 2.50250250e-05, 2.60260260e-05, 2.70270270e-05,
        2.80280280e-05, 2.90290290e-05, 3.00300300e-05, 3.10310310e-05,
        3.20320320e-05, 3.30330330e-05, 3.40340340e-05, 3.50350350e-05,
        3.60360360e-05, 3.70370370e-05, 3.80380380e-05, 3.90390390e-05,
        4.00400400e-05, 4.10410410e-05, 4.20420420e-05, 4.30430430e-05,
        4.40440440e-05, 4.50450450e-05, 4.60460460e-05, 4.70470470e-05,
        4.80480480e-05, 4.90490490e-05, 5.00500501e-05, 5.10510511e-05,
        5.20520521e-05, 5.30530531e-05, 5.40540541e-05, 5.50550551e-05,
        5.60560561e-05, 5.70570571e-05, 5.80580581e-05, 5.90590591e-05,
        6.00600601e-05, 6.10610611e-05, 6.20620621e-05, 6.30630631e-05,
        6.40640641e-05, 6.50650651e-05, 6.60660661e-05, 6.70670671e-05,
        6.80680681e-05, 6.90690691e-05, 7.00700701e-05, 7.10710711e-05,
        7.20720721e-05, 7.30730731e-05, 7.40740741e-05, 7.50750751e-05,
        7.60760761e-05, 7.70770771e-05, 7.80780781e-05, 7.90790791e-05,
        8.00800801e-05, 8.10810811e-05, 8.20820821e-05, 8.30830831e-05,
        8.40840841e-05, 8.50850851e-05, 8.60860861e-05, 8.70870871e-05,
        8.80880881e-05, 8.90890891e-05, 9.00900901e-05, 9.10910911e-05,
        9.20920921e-05, 9.30930931e-05, 9.40941e-05, 9.50950951e-05,
        9.60960961e-05, 9.70970971e-05, 9.80980981e-05, 9.90990991e-05])

    recapture = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.997, 0.982, 0.957,
                            0.912, 0.87, 0.826,
                            0.768, 0.702, 0.656, 0.601, 0.559, 0.514, 0.47, 0.438, 0.404, 0.375, 0.348, 0.323,
                            0.302, 0.277, 0.256, 0.241, 0.219, 0.2, 0.188, 0.175, 0.16, 0.15, 0.138, 0.127,
                            0.122, 0.114, 0.107, 0.098, 0.095, 0.091, 0.082, 0.077, 0.072, 0.066, 0.062, 0.057,
                            0.056, 0.055, 0.054, 0.052, 0.051, 0.049, 0.044, 0.043, 0.042, 0.041, 0.039, 0.036,
                            0.035, 0.035, 0.033, 0.031, 0.029, 0.028, 0.027, 0.027, 0.025, 0.025, 0.023, 0.022,
                            0.02, 0.02, 0.02, 0.02, 0.019, 0.018, 0.017, 0.017, 0.017, 0.016, 0.014, 0.014,
                            0.014, 0.014, 0.014, 0.013, 0.013, 0.013, 0.012, 0.011, 0.011, 0.011, 0.011, 0.011,
                            0.011, 0.01, 0.009, 0.009])

    #best_fit = fit_gs_tweezer_recapture(time, recapture)
    tweezer_recapture_gaussian_process_regress(time, recapture)

    # plt.scatter(time * 1e6, recapture, label='data')
    # plt.plot(time * 1e6, best_fit, label='fit')
    # plt.xlabel('t (us)')
    # plt.ylabel('recapture number')
    # plt.legend(loc='best')
    # # plt.title(f'Trap depth = {trap_temp * 1e3} mK')
    # plt.grid()
    # plt.show()


if __name__ == '__main__':
    # plot_recapture_rate()
    run_fit()
    #plot_loss_landscape()
    
