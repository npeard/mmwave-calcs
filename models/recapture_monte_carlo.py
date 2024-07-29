import numpy as np
from models.utility import wavelength2freq, wavelength2angularfreq
import scipy.constants as const
from scipy.integrate import odeint
from arc import Caesium
import matplotlib.pyplot as plt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

time_data = np.asarray([0.00000000e+00, 1.00100100e-06, 2.00200200e-06,
                        3.00300300e-06,
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
                        9.20920921e-05, 9.30930931e-05, 9.40940941e-05, 9.50950951e-05,
                        9.60960961e-05, 9.70970971e-05, 9.80980981e-05, 9.90990991e-05])
avg_free_recap_data = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.9996, 0.9952,
                                  0.9788, 0.9524, 0.9116,
                                  0.8572, 0.8064, 0.748, 0.6904, 0.6388, 0.5864, 0.5432, 0.5024, 0.458, 0.4232,
                                  0.3892, 0.3612, 0.3328, 0.3076, 0.2828, 0.2608, 0.2416, 0.2236, 0.2068, 0.1944,
                                  0.18, 0.1688, 0.1584, 0.1468, 0.1404, 0.1328, 0.1224, 0.1176, 0.1132, 0.1064,
                                  0.0992, 0.0928, 0.0876, 0.0832, 0.0808, 0.0796, 0.0768, 0.074, 0.0716, 0.0696,
                                  0.0636, 0.0592, 0.0564, 0.0532, 0.0512, 0.0472, 0.0452, 0.0444, 0.0436, 0.0424,
                                  0.042, 0.0412, 0.0392, 0.0384, 0.0372, 0.0356, 0.0328, 0.03, 0.0292, 0.0268,
                                  0.0244, 0.0232, 0.0228, 0.0224, 0.0212, 0.0212, 0.0208, 0.0188, 0.0184, 0.018,
                                  0.0172, 0.016, 0.0156, 0.0152, 0.0136, 0.0136, 0.0136, 0.0124, 0.012, 0.0112,
                                  0.0112, 0.01, 0.0096, 0.0096, 0.0088, 0.008, 0.008, 0.008, 0.008, 0.0076])
std_free_recap_data = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0028,
                                  0.01099818, 0.02241785, 0.03043419, 0.0412, 0.0452566, 0.05534474,
                                  0.06705222, 0.06538991, 0.06816568, 0.07541247, 0.07336048, 0.07762886,
                                  0.07570997, 0.07561587, 0.07654646, 0.07314752, 0.06936973, 0.06473206,
                                  0.06079605, 0.05454686, 0.05212907, 0.04744513, 0.05021713, 0.04699617,
                                  0.04915282, 0.04819295, 0.04896366, 0.04553856, 0.04166341, 0.04303673,
                                  0.04465691, 0.0435688, 0.04536254, 0.04279065, 0.0438105, 0.04228664,
                                  0.04096633, 0.03967064, 0.03938731, 0.03846869, 0.03864919, 0.03714835,
                                  0.0368977, 0.03736094, 0.03386798, 0.03273164, 0.03315177, 0.03139682,
                                  0.0322887, 0.02986905, 0.02961351, 0.0297429, 0.02930938, 0.03010382,
                                  0.03, 0.03030775, 0.03071417, 0.03068289, 0.03046572, 0.0294727,
                                  0.02986905, 0.02807134, 0.02805994, 0.02517459, 0.02201454, 0.02203996,
                                  0.02227465, 0.02140654, 0.01976259, 0.01976259, 0.0187446, 0.01716275,
                                  0.01736203, 0.01708801, 0.016005, 0.016, 0.01614435, 0.01627759,
                                  0.01466424, 0.01466424, 0.01466424, 0.01490772, 0.01496663, 0.01394848,
                                  0.01394848, 0.01341641, 0.0128, 0.0128, 0.01142629, 0.01131371,
                                  0.01131371, 0.01131371, 0.01131371, 0.01123566])


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


def get_trap_freqs(power, waist=1.2e-6, gamma=1 / Caesium().getStateLifetime(
        n=6, l=1, j=1.5), mass=Caesium().mass):
    angular_detuning = (wavelength2angularfreq(1070e-9) -
                        wavelength2angularfreq(852e-9))
    omega_0 = wavelength2angularfreq(852e-9)
    I = 2 * power / (np.pi * waist**2)
    I_sat = 1.6536e1  # 1.6536(15) mW/cm2
    U0 = -const.hbar * gamma / 8 * gamma / angular_detuning * I / I_sat

    # trap temperature
    T = U0 / const.k

    # radial trap frequency
    omega_H = np.sqrt(4 * U0 / (mass * waist**2))

    # longitudinal trap frequency
    zR = np.pi * waist**2 / 1070e-9
    omega_L = np.sqrt(2 * U0 / (mass * zR**2))

    return omega_H, omega_L, T


def monte_carlo_3d(t, omega_H, omega_L, T, N_ensemble=50,
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
    num_shots = 50
    recapture_rate = np.zeros((num_shots, len(t)))
    for n in range(num_shots):
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

        recapture_rate[n, :] = recapture_number / x_ensemble_t.shape[0]

    avg_recapture_rate = np.mean(recapture_rate, axis=0)
    std_recapture_rate = np.std(recapture_rate, axis=0)

    return avg_recapture_rate, std_recapture_rate


def get_simulated_recapture_data(max_us=100e-6, num_recap_times=20, T_ensemble=20e-6):
    t = np.linspace(0, max_us, num_recap_times)
    tweezer_power = np.random.random(1)*50e-3
    omega_H, omega_L, trap_temp = get_trap_freqs(tweezer_power, 1.2e-6)
    recapture_rate, _ = recapture_rate_3d(t, omega_H, omega_L, trap_temp, T_ensemble)
    return t, recapture_rate, tweezer_power


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

    recapture_rate_iHO, std_rate_iHO = recapture_rate_3d(t, omega_H, omega_L,
                                                         trap_temp, T_ensemble, do_iHO=True)
    recapture_rate_free, std_rate_free = recapture_rate_3d(t, omega_H, omega_L,
                                                           trap_temp, T_ensemble, do_iHO=False)

    print(t[::10])
    print(recapture_rate_iHO[::10])
    print(recapture_rate_free[::10])
    print(std_rate_iHO[::10])
    print(std_rate_free[::10])

    plt.errorbar(t * 1e6, recapture_rate_iHO, yerr=std_rate_iHO, label='iHO')
    plt.errorbar(t * 1e6, recapture_rate_free, yerr=std_rate_free, label='free')
    plt.xlabel('t (us)')
    plt.ylabel('recapture number')
    plt.legend(loc='best')
    plt.title(f'Trap depth = {trap_temp * 1e3} mK')
    plt.grid()
    plt.show()


def plot_loss_landscape():
    mse = []
    params = np.linspace(1e-3, 20e-3, 100)

    for p in params:
        rates = []
        omega_H, omega_L, trap_temp = get_trap_freqs(p, 1.2e-6)
        rates = recapture_rate_3d(time_data, omega_H, omega_L,
                                  trap_temp, T_ensemble=20e-6,
                                  do_iHO=False)
        rates = np.asarray(rates)
        mse.append(np.mean((rates - avg_free_recap_data)**2))

    mse = np.asarray(mse)

    plt.plot(params, mse)
    plt.show()


def tweezer_recapture_gaussian_process_regress(recapture_time,
                                               recapture_rate):
    # space = [Real(100e3, 1e6, name="omega_H"),
    #          Real(100e3, 1e6, name="omega_L"),
    #          Real(1e-6, 100e-3, name="trap_temp"),
    #          Real(1e-6, 1e-3, name="T_ensemble")]

    space = [Real(1e-6, 1e-3, name="T_ensemble"),
             Real(1e-3, 100e-3, name="power")]

    # True values for data set
    # omega_H = 0.8e6  # Hz
    # omega_L = 0.19e6  # Hz
    # trap_temp = 2.63e-3  # K
    # T_ensemble = 20e-6  # 168e-6 #K

    @use_named_args(space)
    def objective_func(T_ensemble, power):
        omega_H, omega_L, trap_temp = get_trap_freqs(power, waist=1.2e-6)
        predicted_rate, _ = recapture_rate_3d(recapture_time,
                                              omega_H, omega_L,
                                              trap_temp, T_ensemble, do_iHO=False)
        loss = np.mean((recapture_rate - predicted_rate)**2)
        return loss

    result = gp_minimize(objective_func, space, n_calls=100,
                         acq_func="gp_hedge",
                         n_random_starts=25, random_state=0)

    # result = forest_minimize(objective_func, space, n_calls=200,
    #                          base_estimator="ET", random_state=0)

    print('result params:',result.x)
    print('result freqs:', get_trap_freqs(result.x[1], 1.2e-6))
    print(result.fun)

    plot_convergence(result, yscale="log")
    plt.show()

    # plot_objective(result, n_points=10)
    # plt.show()
    omega_H, omega_L, trap_temp = get_trap_freqs(result.x[1], 1.2e-6)
    best_fit, error = (recapture_rate_3d(recapture_time, omega_H, omega_L,
                                         trap_temp, result.x[0], do_iHO=False))

    plt.scatter(recapture_time * 1e6, recapture_rate, label='data')
    plt.plot(recapture_time * 1e6, best_fit, label='GPR fit')
    plt.xlabel('t (us)')
    plt.ylabel('recapture rate')
    plt.legend(loc='best')
    # plt.title(f'Trap depth = {trap_temp * 1e3} mK')
    plt.grid()
    plt.show()


def run_fit():

    # best_fit = fit_gs_tweezer_recapture(time, recapture)
    tweezer_recapture_gaussian_process_regress(time_data, avg_free_recap_data)

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
    
    recap_times, recap_rate, power = get_simulated_recapture_data(50e-6, 40,
                                                                12e-6)
    print("tweezer power = ", power)
    print("frequencies and trap temp = ", get_trap_freqs(power, 1.2e-6))
    tweezer_recapture_gaussian_process_regress(recap_times, recap_rate)
