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


def iHO_a_3d(x, y, z, omega_H, omega_L):
    """
    Calculate the acceleration components in the iHO frame for a given position and frequencies.

    Parameters
    ----------
    x : float
        x-coordinate in the iHO frame.
    y : float
        y-coordinate in the iHO frame.
    z : float
        z-coordinate in the iHO frame.
    omega_H : float
        radial trap angular frequency
    omega_L : float
        longitudinal trap angular frequency

    Returns
    -------
    ax : float
        x-component of the acceleration in the iHO frame.
    ay : float
        y-component of the acceleration in the iHO frame.
    az : float
        z-component of the acceleration in the iHO frame.
    """
    ax = (omega_H**2) * x
    ay = (omega_H**2) * y
    g = 9.8  # m/s^2 gravity
    az = (omega_L**2) * z - g
    return ax, ay, az


def newton_3d(f, t, omega_H, omega_L):
    """
    Calculate the next position and velocity of a particle in 3D space
    using Newton's second law of motion.

    Parameters
    ----------
    f : list
        List containing the current position and velocity components of the particle.
        [x, vx, y, vy, z, vz]
    omega_H : float
        Radial trap angular frequency
    omega_L : float
        Longitudinal trap angular frequency

    Returns
    -------
    dfdt : list
        List containing the next position and velocity components of the particle.
        [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = f
    ax, ay, az = iHO_a_3d(x, y, z, omega_H, omega_L)
    dfdt = [vx, vy, vz, ax, ay, az]
    return dfdt


def get_trap_depth(power, waist, gamma=1 / Caesium().getStateLifetime(
        n=6, l=1, j=1.5)):
    angular_detuning = (wavelength2angularfreq(1070e-9) -
                        wavelength2angularfreq(852e-9))
    intensity = 2 * power / (np.pi * waist**2)
    I_sat = 1.6536e1  # 1.6536(15) mW/cm2
    U0 = -const.hbar * gamma / 8 * gamma / angular_detuning * intensity / I_sat

    return U0


def get_trap_freqs(power, waist=1.15e-6, mass=Caesium().mass):
    """
    Calculate the trap frequencies based on the trap parameters.

    Parameters
    ----------
    power : float
        Power of the trap in mW.
    waist : float, optional
        Waist of the laser beam in m. Default is 1.2 microns.
    gamma : float, optional
        Laser-atom interaction parameter in s^-1. Default is calculated from the
        6s-1/2 state of caesium.
    mass : float, optional
        Mass of the trap particles in kg. Default is the mass of caesium.

    Returns
    -------
    omega_H : float
        Radial trap angular frequency in rad/s.
    omega_L : float
        Longitudinal trap angular frequency in rad/s.
    trap_temperature : float
        Trap temperature in K.
    """

    U0 = get_trap_depth(power, waist)

    # trap temperature
    trap_temperature = U0 / const.k

    # radial trap frequency
    omega_H = np.sqrt(4 * U0 / (mass * waist**2))

    # longitudinal trap frequency
    zR = np.pi * waist**2 / 1070e-9
    omega_L = np.sqrt(2 * U0 / (mass * zR**2))

    return omega_H, omega_L, trap_temperature


def get_tau(power, T_ensemble, waist=1.15e-6):
    U0 = get_trap_depth(power, waist)
    tau = const.k * T_ensemble / U0

    return tau


def radial_recapture_rate(t, power, T_ensemble, waist=1.15e-6):
    # model courtesy of Monika Schleier-Smith
    tau = get_tau(power, T_ensemble, waist)
    omega_H, _, _ = get_trap_freqs(power, waist)

    probability = 1 - tau**(1 / (tau * (1 + omega_H**2 * t**2)))

    return probability


def monte_carlo_3d(t, omega_H, omega_L, T_ensemble, N_ensemble=50,
                   mass=Caesium().mass, do_iHO=False):
    """
    Generate ensembles of 3D positions and velocities using a Monte Carlo method.

    Parameters
    ----------
    t : array-like
        Time points at which to generate ensembles.
    omega_H : float
        Radial trap angular frequency in rad/s.
    omega_L : float
        Longitudinal trap angular frequency in rad/s.
    T_ensemble : float
        Ensemble temperature in K.
    N_ensemble : int, optional
        Number of atoms in ensemble to generate. Default is 50.
    mass : float, optional
        Mass of the trap particles in kg. Default is the mass of caesium.
    do_iHO : bool, optional
        If True, generate ensembles for the iHO instead of free space.
        Default is False.

    Returns
    -------
    x_ensemble : ndarray
        Array of shape (N_ensemble, len(t)) containing ensembles of x positions.
    vx_ensemble : ndarray
        Array of shape (N_ensemble, len(t)) containing ensembles of x velocities.
    y_ensemble : ndarray
        Array of shape (N_ensemble, len(t)) containing ensembles of y positions.
    vy_ensemble : ndarray
        Array of shape (N_ensemble, len(t)) containing ensembles of y velocities.
    z_ensemble : ndarray
        Array of shape (N_ensemble, len(t)) containing ensembles of z positions.
    vz_ensemble : ndarray
        Array of shape (N_ensemble, len(t)) containing ensembles of z velocities.
    """

    delta_r = np.sqrt(const.k * T_ensemble / (mass * np.array([omega_H, omega_H, omega_L])**2))
    delta_v = np.sqrt(const.k * T_ensemble / mass) * np.ones(3)
    x_ensemble, y_ensemble, z_ensemble = (delta_r.T * np.random.normal(0, 1, size=(N_ensemble, 3))).T
    vx_ensemble, vy_ensemble, vz_ensemble = (delta_v.T * np.random.normal(0, 1, size=(N_ensemble, 3))).T

    f = wavelength2freq(1070e-9)
    f0 = wavelength2freq(852e-9)
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


def energy_3d(x, vx, y, vy, z, vz, mass, power, waist):
    """
    Calculate the energy of a 3D system of atoms.

    Parameters
    ----------
    x : array-like
        Array of x positions of atoms.
    vx : array-like
        Array of x velocities of atoms.
    y : array-like
        Array of y positions of atoms.
    vy : array-like
        Array of y velocities of atoms.
    z : array-like
        Array of z positions of atoms.
    vz : array-like
        Array of z velocities of atoms.
    mass : float
        Mass of atoms in kg.
    omega_H : float
        Radial trap angular frequency in rad/s.
    omega_L : float
        Longitudinal trap angular frequency in rad/s.

    Returns
    -------
    E : float
        Total energy of the system in J.
    """
    omega_H, omega_L, _ = get_trap_freqs(power, waist)

    # calculate energy of system
    E = 0.5 * mass * (vx**2 + vy**2 + vz**2) + 0.5 * mass * (omega_H**2 * (
        x**2 + y**2) + omega_L**2 * (z**2))
    return E


def total_energy(x, vx, y, vy, z, vz, mass, power, waist):
    """
    Calculate the total energy of a 3D system of atoms.

    Parameters
    ----------
    x : array-like
        Array of x positions of atoms.
    vx : array-like
        Array of x velocities of atoms.
    y : array-like
        Array of y positions of atoms.
    vy : array-like
        Array of y velocities of atoms.
    z : array-like
        Array of z positions of atoms.
    vz : array-like
        Array of z velocities of atoms.
    mass : float
        Mass of atoms in kg.
    power : float
        Power of the laser beam in W.
    waist : float
        Waist of the laser beam in m.

    Returns
    -------
    E : float
        Total energy of the system in J.
    """
    zR = np.pi * waist**2 / 1070e-9
    U0 = get_trap_depth(power, waist)
    # radial trap frequency
    omega_H = np.sqrt(4 * U0 / (mass * waist**2))

    potential = -U0 * np.exp(-2 * (x**2 + y**2) / waist**2) / (1 + z**2 / zR**2)

    # calculate energy of system
    E = 0.5 * mass * (vx**2 + vy**2 + vz**2) + potential
    return E


def recapture_rate_3d(t, power, waist, T_ensemble, mass=Caesium().mass,
                      do_iHO=False):
    """
    Calculate the recapture rate of a 3D system of atoms.

    Parameters
    ----------
    t : array-like
        Array of times in seconds.
    omega_H : float
        Radial trap angular frequency in rad/s.
    omega_L : float
        Longitudinal trap angular frequency in rad/s.
    trap_temp : float
        Trap temperature in K.
    T_ensemble : array-like
        3D array of x, y, z positions and velocities of atoms.
    mass : float, optional
        Mass of atoms in kg.
    do_iHO : bool, optional
        Flag to apply iHO conditions, by default False.

    Returns
    -------
    t : array-like
        Array of times in seconds.
    recapture_rate : array-like
        1D array of recapture rates for each time in t.
    std_recapture_rate : array-like
        1D array of standard deviation of recapture rates for each time in t.
    """
    omega_H, omega_L, trap_temperature = get_trap_freqs(power, waist)

    effective_trap_depth = const.k * trap_temperature
    num_shots = 50
    recapture_rate = np.zeros((num_shots, len(t)))
    for n in range(num_shots):
        (x_ensemble_t,
         vx_ensemble_t,
         y_ensemble,
         vy_ensemble,
         z_ensemble,
         vz_ensemble) = monte_carlo_3d(t, omega_H, omega_L,
                                       T_ensemble=T_ensemble, do_iHO=do_iHO)

        # energy_ind = energy_3d(x_ensemble_t, vx_ensemble_t, y_ensemble, vy_ensemble, z_ensemble, vz_ensemble, mass, omega_H, omega_L)
        energy_ind = total_energy(x_ensemble_t, vx_ensemble_t, y_ensemble,
                                  vy_ensemble, z_ensemble, vz_ensemble, mass,
                                  power, waist)

        recaptured = np.zeros_like(energy_ind)
        recaptured[energy_ind <= 0] = 1

        recapture_number = np.sum(recaptured, axis=0)

        recapture_rate[n, :] = recapture_number / x_ensemble_t.shape[0]

    avg_recapture_rate = np.mean(recapture_rate, axis=0)
    std_recapture_rate = np.std(recapture_rate, axis=0)

    return avg_recapture_rate, std_recapture_rate


def get_simulated_recapture_data(max_us, num_recap_times, T_ensemble):
    """
    Generate simulated recapture data for a given duration and number of time steps.

    Parameters
    ----------
    max_us : float, optional
        Maximum time in microseconds. Default is 100e-6.
    num_recap_times : int, optional
        Number of time steps. Default is 20.
    T_ensemble : float, optional
        Ensemble temperature in Kelvin. Default is 20e-6.

    Returns
    -------
    t : ndarray
        Array of times in seconds.
    recapture_rate : ndarray
        2D array of recapture rates for each time and ensemble shot.
    tweezer_power : ndarray
        Array of tweezer powers used to generate the data.
    """
    t = np.linspace(0, max_us, num_recap_times)
    tweezer_power = np.random.random(1) * 10e-3
    recapture_rate, recapture_std = recapture_rate_3d(t, tweezer_power,
                                                      1.15e-6, T_ensemble)
    return t, recapture_rate, recapture_std, tweezer_power


def plot_recapture_rate():
    omega_H = 0.8e6  # Hz
    omega_L = 0.19e6  # Hz
    T_ensemble = 20e-6  # 168e-6 #K
    trap_temp = 2.63e-3  # K
    t = np.linspace(0, 100, 1000) * 1e-6

    power = 2.5e-3  # W
    waist = 1.15e-6  # m

    recapture_rate_iHO, std_rate_iHO = recapture_rate_3d(t, power,
                                                         waist, T_ensemble,
                                                         do_iHO=True)
    recapture_rate_free, std_rate_free = recapture_rate_3d(t, power, waist,
                                                           T_ensemble, do_iHO=False)

    plt.errorbar(t * 1e6, recapture_rate_iHO, yerr=std_rate_iHO, label='iHO')
    plt.errorbar(t * 1e6, recapture_rate_free, yerr=std_rate_free, label='free')
    plt.xlabel('t (us)')
    plt.ylabel('recapture number')
    plt.legend(loc='best')
    plt.title(f'Trap depth = {trap_temp * 1e3} mK')
    plt.grid()
    plt.show()


def plot_loss_landscape(recapture_time, recapture_rate,
                        tweezer_power, tweezer_waist=1.15e-6):
    temperatures = np.linspace(1e-6, 20e-6, 100)
    losses = []
    for T in temperatures:
        rate = radial_recapture_rate(recapture_time, tweezer_power, T)
        loss = np.mean((rate - recapture_rate)**2)
        losses.append(loss)

    plt.plot(temperatures, np.asarray(losses))
    plt.xlabel('T (K)')
    plt.ylabel('losses')
    plt.grid()
    plt.show()


def tweezer_temperature_regress(recapture_time, recapture_rate,
                                tweezer_power, recapture_std=None,
                                tweezer_waist=1.15e-6,
                                plot_results=False, quantify_error=False):
    """
    Returns the ensemble temperature that best fits the given recapture rate data.

    Parameters
    ----------
    recapture_time : ndarray
        Array of times in seconds.
    recapture_rate : ndarray
        Array of recapture rates for each time.
    recapture_std : ndarray
        Array of standard deviations of recapture rates for each time.
    tweezer_power : float
        Tweezer power used to generate the data in W.
    tweezer_waist : float, optional
        Tweezer waist size in meters. Default is 1.15e-6.
    plot_results : bool, optional
        If True, plots the data and the best fit. Default is False.
    quantify_error : bool, optional
        If True, randomly initializes regression. Default is False.

    Returns
    -------
    T_ensemble : float
        Ensemble temperature in Kelvin.
    """

    space = [Real(1e-6, 1e-3, name="T_ensemble")]

    @use_named_args(space)
    def objective_func(T_ensemble):
        predicted_rate, _ = recapture_rate_3d(recapture_time,
                                              tweezer_power, tweezer_waist, T_ensemble, do_iHO=False)
        loss = np.mean((recapture_rate - predicted_rate)**2)
        return loss

    if quantify_error:
        random_state = None
    else:
        random_state = 0

    # use skopt GPR to fit 3d recapture model
    result = gp_minimize(objective_func, space, n_calls=100,
                         acq_func="gp_hedge",
                         n_random_starts=25, random_state=random_state)

    # use lmfit to fit radial recapture model
    from lmfit import Model, Parameters
    params = Parameters()
    params.add("T_ensemble", value=12e-6, min=1e-6, max=5e-3)
    params.add("power", value=tweezer_power, vary=False)
    params.add("waist", value=1.15e-6, vary=False)
    radial_model = Model(radial_recapture_rate, independent_vars=["t"])

    radial_result = radial_model.fit(recapture_rate, params,
                                     t=recapture_time)

    # result = forest_minimize(objective_func, space, n_calls=200,
    #                          base_estimator="ET", random_state=0)

    # print("Best score: ", result.fun)
    # print("Best parameters: ", result.x)

    # plot_convergence(result, yscale="log")
    # plt.show()

    # plot_objective(result, n_points=10)
    # plt.show()

    if plot_results:
        fine_grid = np.linspace(recapture_time[0], recapture_time[-1], 1000)
        best_fit, error = (recapture_rate_3d(fine_grid, tweezer_power,
                                             tweezer_waist,
                                             result.x[0],
                                             do_iHO=False))

        if recapture_std is not None:
            plt.errorbar(recapture_time * 1e6, recapture_rate,
                         yerr=recapture_std, label='data', color='black')
        else:
            plt.scatter(recapture_time * 1e6, recapture_rate, label='data',
                        color='black')

        # GPR model result
        plt.plot(fine_grid * 1e6, best_fit, label='GPR fit, T = {:.1f} uK'.format(result.x[0] * 1e6), color='blue')
        plt.fill_between(fine_grid * 1e6, best_fit - error, best_fit + error,
                         alpha=0.2, color='blue')

        # radial model result
        best_radial_fit = radial_recapture_rate(fine_grid, tweezer_power,
                                                radial_result.params["T_ensemble"].value)
        sigma = np.sqrt(radial_result.covar[0, 0])
        upper_sigma = radial_recapture_rate(fine_grid, tweezer_power,
                                            radial_result.params[
                                                "T_ensemble"].value + sigma)
        lower_sigma = radial_recapture_rate(fine_grid, tweezer_power,
                                            radial_result.params[
                                                "T_ensemble"].value - sigma)
        plt.plot(
            fine_grid * 1e6,
            best_radial_fit,
            label='radial fit, T = {:.1f} uK'.format(
                radial_result.params["T_ensemble"].value * 1e6),
            color='orange')
        plt.fill_between(fine_grid * 1e6, lower_sigma, upper_sigma,
                         alpha=0.2, color='orange')

        plt.xlabel('t [us]')
        plt.ylabel('recapture rate')
        plt.legend(loc='best')
        omega_H, omega_L, trap_temperature = get_trap_freqs(tweezer_power,
                                                            tweezer_waist)
        plt.title(f'Trap depth = {trap_temperature * 1e3} mK, Tweezer power = {tweezer_power * 1e3} mW')
        plt.grid()
        plt.show()

    T_ensemble = result.x[0]

    return T_ensemble


def run_model(recapture_time, recapture_rate, tweezer_power, recapture_std=None,
              quantify_error=False, true_T_ensemble=None):
    """
    Runs the model to estimate the ensemble temperature.

    Parameters
    ----------
    recapture_time : ndarray
        Array of times in seconds.
    recapture_rate : ndarray
        Array of recapture rates for each time and ensemble shot.
    tweezer_power : float
        Tweezer power used to generate the data in W.
    recapture_std : ndarray, optional
        Array of standard deviations of the recapture rates.
    quantify_error : bool, optional
        If True, randomly initializes regression. Default is False.
    true_T_ensemble : float, optional
        True ensemble temperature in Kelvin, if known.

    Returns
    -------
    """

    if quantify_error:
        num_shots = 10
        temperatures = []
        for i in np.arange(num_shots):
            T_ensemble = tweezer_temperature_regress(recapture_time,
                                                     recapture_rate,
                                                     tweezer_power,
                                                     quantify_error=True)
            temperatures.append(T_ensemble)
        temperatures = np.asarray(temperatures)
        print("T_ensemble [uK]= ", np.mean(temperatures * 1e6), "+/-", np.std(
            temperatures * 1e6))

        plt.hist(temperatures * 1e6)
        plt.xlabel('Temperature [uK]')
        plt.ylabel('Number of shots')
        if true_T_ensemble is not None:
            plt.axvline(x=true_T_ensemble * 1e6, color='r', linestyle='--',
                        label='True T_ensemble')
        plt.legend(loc='best')
        plt.show()

    else:
        T_ensemble = tweezer_temperature_regress(recapture_time, recapture_rate,
                                                 tweezer_power, recapture_std=recapture_std,
                                                 plot_results=True)
        print("T_ensemble [uK] = ", T_ensemble * 1e6)


if __name__ == '__main__':
    # plot_recapture_rate()

    recap_times, recap_rate, recap_std, tweezer_power = (
        get_simulated_recapture_data(50e-6, 10, 5e-6))
    print("tweezer power = ", tweezer_power)
    run_model(recap_times, recap_rate, tweezer_power, recapture_std=recap_std,
              quantify_error=False,
              true_T_ensemble=5e-6)

    # plot_loss_landscape(recap_times, recap_rate, tweezer_power)
