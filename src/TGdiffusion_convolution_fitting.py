"""

Module to parse transport data from TG and compare
with a convolution-based solution of the diffusion equation.

Author: Peter Bork
March 2022
"""

import numpy as np
from scipy import interpolate, integrate
from scipy.optimize import minimize_scalar
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

def create_diff_eq_solution_function(h, t):
    """ Creates solution function for diffusion problem.

    Args:
        h: 1D numpy array (floats) length N of u(0, t) at times given in t
        t: 1D numpy array (floats) length N of timepoints for measurements in h
    Returns:
        python function u(x, t, D), which solves the problem.
    """

    # interpolate h
    H_function = interpolate.interp1d(t, h)

    def psi(x, t, D):
        if t == 0:
            return 0

        num = x * np.exp(-np.square(x) / (4 * D * t))
        denom = np.sqrt(4 * np.pi * D * t**3)

        return num / denom

    def u(x, T, D):
        integrand = lambda s: psi(x, T-s, D) * H_function(s)
        y, err = integrate.quad_vec(integrand, t[0], T)
        y = np.where(x != 0, y, H_function(T))

        return y

    return u

def create_diff_eq_solution_function_of_D_h_t(D, h, t):
    """ Creates solution function for diffusion problem.

    Args:
        D: Diffusion coefficient
        h: 1D numpy array (floats) length N of u(0, t) at times given in t
        t: 1D numpy array (floats) length N of timepoints for measurements in h
    Returns:
        python function u(x, t), which solves the problem.
    """

    # interpolate h
    H_function = interpolate.interp1d(t, h)

    def psi(x, t):
        if t == 0:
            return 0

        num = x * np.exp(-np.square(x) / (4 * D * t))
        denom = np.sqrt(4 * np.pi * D * t**3)

        return num / denom

    def u(x, T):
        integrand = lambda s: psi(x, T-s) * H_function(s)
        y, err = integrate.quad_vec(integrand, t[0], T)
        y = np.where(x != 0, y, H_function(T))

        return y

    return u

def make_prediction_func(
    source_vector,
    conc_matrix,
    time_vector,
    distance_vector,
):
    diffusion_solution_func = create_diff_eq_solution_function(source_vector, time_vector)
    
    prediction_matrix = np.nan * np.zeros_like(conc_matrix)
    
    def prediction_func(D_guess):
        for row_index in range(prediction_matrix.shape[0]):
            prediction_matrix[row_index, :] = diffusion_solution_func(
                distance_vector,
                time_vector[row_index],
                D_guess
            )
        
        return prediction_matrix
    
    return prediction_func


def make_error_function(
    source_vector,
    conc_matrix,
    time_vector,
    distance_vector,
    lam = 0.0,
):
    prediction_func = make_prediction_func(source_vector, conc_matrix, time_vector, distance_vector)
    
    def error_func(D_guess):
        prediction = prediction_func(D_guess)
        error = np.sqrt(np.mean(np.square(prediction - conc_matrix))) + lam * D_guess
        return error
    
    return error_func


def rollingmeandownsample(matrix_or_vector, windowsize):
    df = pd.DataFrame(matrix_or_vector)
    return df.rolling(windowsize).mean().values[::windowsize]


def find_RMS_minimizing_D_for_data_in_folder(folder, lam, subsampleskipsteps, maxiter):
    
    identifier = folder.split("/")[-1]
    
    source_vector = np.load(folder + "/source_vector.npy")
    conc_matrix = np.load(folder + "/concmatrix.npy")
    time_vector = np.load(folder + "/time_vector.npy")
    distance_vector = np.load(folder + "/distance_vector.npy")
    
    #source_vector = rollingmeandownsample(source_vector, subsampleskipsteps)
    #conc_matrix = rollingmeandownsample(conc_matrix, subsampleskipsteps)
    #time_vector = rollingmeandownsample(time_vector, subsampleskipsteps)
    source_vector = source_vector[::subsampleskipsteps]
    conc_matrix = conc_matrix[::subsampleskipsteps, :]
    time_vector = time_vector[::subsampleskipsteps]
    
    conc_timesteps, conc_distances = conc_matrix.shape
    assert distance_vector.size == conc_distances
    assert time_vector.size == conc_timesteps
    assert time_vector.size == source_vector.size
    
    error_func = make_error_function(
        source_vector,
        conc_matrix,
        time_vector,
        distance_vector,
        lam,
    )
    
    #minresult = minimize_scalar(
    #    error_func,
    #    bounds=(0.000001, np.Inf),
    #    method='bounded',
    #    options={'maxiter': maxiter},
    #)
    minresult = minimize_scalar(
        error_func,
        method='Golden',
    )
    best_D = minresult.x
    best_error = minresult.fun
    
    return best_D, best_error, identifier, minresult

def plot_compare_data_prediction(folder, identifier, D_guess, subsampleskipsteps, saveto):
        
    source_vector = np.load(folder + "/source_vector.npy")
    conc_matrix = np.load(folder + "/concmatrix.npy")
    time_vector = np.load(folder + "/time_vector.npy")
    distance_vector = np.load(folder + "/distance_vector.npy")
    
    source_vector = source_vector[0::subsampleskipsteps]
    conc_matrix = conc_matrix[0::subsampleskipsteps, :]
    time_vector = time_vector[0::subsampleskipsteps]

    prediction_func = make_prediction_func(
        source_vector,
        conc_matrix,
        time_vector,
        distance_vector,
    )
    
    prediction = prediction_func(D_guess)
    
    num_rois = conc_matrix.shape[1]
    #for i in range(num_rois):
    #plt.plot(time_vector, source_vector, label="source")
    #plt.plot(time_vector, prediction[:, 0], linestyle=':', label="pred")
    #plt.plot(time_vector, conc_matrix[:, 0], linestyle='-.', label="data")
    #plt.legend()
    #plt.show()
    
    fig, axes = plt.subplots(num_rois, 1, sharex=True, figsize=(6, 12))
    for i, ax in enumerate(axes):
        axes[i].plot(time_vector, conc_matrix[:, i], marker='o', label="data")
        axes[i].plot(time_vector, prediction[:, i], label="prediction")
    
    axes[0].legend()
    axes[-1].set_xlabel("time (h)")
    axes[0].set_ylabel("conc.")
    axes[0].set_title(identifier)
    plt.savefig(saveto, bbox_inches='tight')
    
    plt.show()
