import time

import numpy as np
from scipy.signal import hilbert

def forced_hkb_ode(x, stim, F=1.0, gamma=1, delta=1, epsi=-0.7, omega=1.5):
    """
    Defines the system of ODEs for the forced HKB model.

    This function integrates the forced HKB model, which consists of two-dimensional oscillator with a Rayleigh term, 
    a Van der Pol term, a negative linear damping term, and a forcing term. The system is non-autonomous because the 
    time is explicitly used for the forcing which makes the system three-dimensional in total, hence the third state.

    References:
    ----------
    Fuchs, A., Jirsa, V. K., Haken, H., & Kelso, J. A. S. (1996). Extending the HKB-Model of coordinated movement to 
        oscillators with different eigenfrequencies. Biological Cybernetics 74, 21-30. https://doi.org/10.1007/BF00199134
    Kelso, J. A. S. (2008). Haken-Kelso-Bunz model. Scholarpedia, 3(10), 1612. https://doi.org/10.4249/scholarpedia.1612
    
    Parameters:
    ----------
    x : array-like
        State variables [x1, x2, x3], where:
        x1 is the velocity,
        x2 is the position,
        x3 is the time variable.
    F : float, optional (default: 1.0)
        Forcing amplitude.
    stim : float
        Forcing term.
    gamma : float, optional (default: 1)
        VDP term.
    delta : float, optional (default: 1)
        Rayleigh term.
    epsi : float, optional (default: -0.7)
        Negative linear damping.
    omega : float, optional (default: 1.5)
        Intrinsic frequency of the oscillator.
    
    Returns:
    -------
    xdot : ndarray
        Derivatives [dx1/dt, dx2/dt, dx3/dt].
    """
    
    # State variables
    x1, x2, x3 = x
    
    # System of equations
    dx1 = - delta * x1**3 - (omega**2) * x2 - gamma * x1 * x2**2 - epsi * x1 + F * stim
    dx2 = x1
    dx3 = 1
    
    return np.array([dx1, dx2, dx3])


def simulate_forced_hkb_grid_params(
    coupling_strengths_list, 
    frequencies_list, 
    dt=0.01, 
    iters=10000,
    noise_strength = 0.6,
    omega_0 = 2, 
    initial_conditions=None
    ):
    """
    Simulates the forced HKB model for a grid of coupling strengths and frequencies.
    
    This function integrates the forced HKB model using the Euler method for a specified number of iterations and
    time step. It computes the time series of the first dimension of the state vector, the relative phase of the
    oscillator with respect to the forcing, and statistics of the relative phase (dispersion and mean).

    References:
    ----------
    Higham., D. J. (2001). An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations. 
        SIAM Review, 43(3), 525–546. https://doi.org/10.1137/S0036144500378302
    Mardia, K. V. (1972). Statistics of Directional Data. Academic Press, London.

    Parameters:
    ----------
    coupling_strengths_list : list or array
        List or array of coupling strengths to simulate.
    frequencies_list : list or array
        List or array of frequencies to simulate.
    dt : float, optional
        Time step for the simulation (default is 0.01).
    iters : int, optional
        Number of iterations for the simulation (default is 10000).
    noise_strength : float, optional
        Strength of the noise applied to the first dimension of the state vector (default is 0.06).
    omega_0 : float, optional
        Natural frequency of the oscillator (default is 2).
    initial_conditions : array, optional
        Initial conditions for the state vector. If not provided, defaults to [[0, -0.3, 0]].

    Returns:
    -------
    x_list : array
        Time series of the first dimension of the state vector for each coupling strength and frequency.
    phi_rel_list : array
        Relative phase of the oscillator with respect to the forcing for each coupling strength and frequency.
    stats_list : array
        Statistics of the relative phase (dispersion and mean) for each coupling strength and frequency.
    """

    # Time vector
    t = np.arange(0, iters * dt, dt)

    # Noise vector: applied only to the first dimension of the state vector since the other two dimensions define the 
    # second-order derivative and the time
    D = [noise_strength, 0, 0]

    # Initial conditions: if not provided, use a default value
    if initial_conditions is None:
        initial_conditions = np.array([[0, -0.3, 0]], order='C')

    # Empty array to store time series
    x_list = np.empty((iters, np.size(coupling_strengths_list), np.size(frequencies_list)))
    phi_rel_list = np.empty((iters, np.size(coupling_strengths_list), np.size(frequencies_list)))
    stats_list = np.empty((2, np.size(coupling_strengths_list), np.size(frequencies_list)))

    # Calculate the total number of simulations to inform the progress bar
    total_simulations = np.size(coupling_strengths_list) * np.size(frequencies_list)
    simulation_idx, last_simulation_percentage = 0, 0

    # Measure the time taken for the simulation of one frequency set
    start_time = time.time()

    # Loop through frequencies and coupling strengths
    # This ordering saves simulation time by avoiding repeated definition of the forcing time series
    for freq_idx, freq in enumerate(frequencies_list):

        # Define forcing time series and its instantaneous phase
        # Instantaneous phase is used at the end of the loop to calculate the relative phase
        forcing_series = np.sin(freq * t)
        phi_stim = np.angle(hilbert(forcing_series))

        # Inform the user about the time estimated for the simulation
        if freq_idx == 1:
            loop_time = time.time()
            time_minutes = (loop_time - start_time) / 60
            print(f"Time taken for the first frequency set: {time_minutes:.2f} minutes")
            print(f"Estimated time for the entire simulation: {(time_minutes) * np.size(frequencies_list):.2f} minutes")

        for F_idx, F in enumerate(coupling_strengths_list):

            # Inform the user of the current simulation progress
            simulation_percentage = (simulation_idx / total_simulations) * 100
            if simulation_percentage - last_simulation_percentage >= 10:
                stage_time = time.time() - start_time
                print(f"Simulation progress: {simulation_percentage:n}% (time elapsed: {stage_time / 60:.2f} minutes)")
                last_simulation_percentage = last_simulation_percentage + 10
        
            # Integrate the system with Euler method
            x = np.array(initial_conditions)

            for i in range(iters-1):
                forcing = forcing_series[i]
                xdot = forced_hkb_ode(x[-1], forcing, F=F, omega=omega_0)
                x = np.append(x, 
                              np.array([x[-1] + (xdot + np.sqrt(D) * np.random.normal(0, 1, 3)) * dt]), 
                              axis = 0)
            
            # Calculate the instantaneous phase of the oscillator, then the relative phase
            # By convention, phi_rel < 0 means the oscillator is lagging behind the forcing and phi_rel > 0 means it is leading
            phi_x = np.angle(hilbert(x[:,0]))
            phi_rel = phi_x - phi_stim
            phi_rel = np.mod(phi_rel + np.pi, 2 * np.pi) - np.pi  # Normalize the relative phase to the range [-pi, pi]

            # Calculate the dispersion of the relative phase and its mean value (circular statistics; see Mardia, 1972)
            # Trim the first 10% and last 10% of the time series to avoid edge effects 
            phi_rel_trimmed = phi_rel[int(iters * 0.1):int(iters * 0.9)]
            resultant_vector = np.mean(np.exp(1j * phi_rel_trimmed))
            phi_rel_mean = np.angle(resultant_vector)
            phi_rel_dispersion = np.sqrt(-2 * np.log(np.abs(resultant_vector)))

            # Store the time series
            x_list[:, F_idx, freq_idx] = x[:, 0]
            phi_rel_list[:, F_idx, freq_idx] = phi_rel
            stats_list[:, F_idx, freq_idx] = phi_rel_dispersion, phi_rel_mean

            simulation_idx += 1
    
    # Print the final simulation time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total simulation time: {total_time / 60:.2f} minutes")
    
    return t, x_list, phi_rel_list, stats_list