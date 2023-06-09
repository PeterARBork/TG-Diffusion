"""

Module to parse transport data from cochlear aqueduct and cochlea and compare
with a convolution-based solution of the diffusion equation.

Author: Peter Bork
March 2022
"""

import numpy as np
from scipy import interpolate, integrate
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

def plot_raw_data_samples(spacetime, distances, skipsteps):
    for i in range(0, len(spacetime), skipsteps):
        plt.plot(distances, spacetime[i, :], marker='o')

    plt.xlabel('distance [mm]')
    plt.ylabel('MR signal')
    plt.title(f'Raw data every {skipsteps}th frame')
    plt.show()

def plot_data_vs_model(tac_df, distances, data_spacetime, prediction_spacetime,
                        mouse_id=''):
    xs = np.linspace(distances[0], distances[-1], 20)

    t_hat = tac_df.loc[:, 'time'].values
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, distance_i in enumerate(distances):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        ax.plot(tac_df['time'], data_spacetime[:, i], label='data', marker='x', color='darkgreen')

        #pred = np.array([u_func(distance_i, t, D_guess) for t in t_hat])
        #ax.plot(t_hat, pred, label='prediction', marker='o')
        ax.plot(t_hat, prediction_spacetime[:, i], label='prediction', marker='o')

        midstr = mouse_id + ', ' if len(mouse_id) > 0 else ''
        ax.set_title(midstr + f'x = {distance_i:.2f} mm')
        ax.set_xlabel('time [h]')
        ax.set_ylabel('concentration')
        ax.legend()

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()

def run_Cochlear_aqueduct_analysis(tac_df, dist_df, L_or_R, D_guess, skipsteps, show_raw_data=True, mouse_id=''):

    OCA_dists = create_direct_distance_vector(
        dist_df,
        'Cochlear aqueduct',
        'Outside cochlear aqueduct ' + L_or_R,
        L_or_R,
        4
    )

    measurement_ROIs  = [
        f'Outside cochlear aqueduct {L_or_R} mean',
        f'Cochlear aqueduct 1 {L_or_R} mean',
        f'Cochlear aqueduct 2 {L_or_R} mean',
        f'Cochlear aqueduct 3 {L_or_R} mean',
    ]

    (diffusion_line_spacetime,
     predictions_spacetime,
     u_func) = compare_data_to_model(tac_df, measurement_ROIs, OCA_dists, D_guess, skipsteps, mouse_id, show_raw_data)

    return (diffusion_line_spacetime, predictions_spacetime, u_func)

def run_Cochlea_analysis(tac_df, dist_df, L_or_R, D_guess, skipsteps, show_raw_data=True, mouse_id=''):

    C_dists = create_spiral_distance_vector(dist_df, 'Cochlear', L_or_R, 4)

    measurement_ROIs  = [
        f'Cochlear 1 {L_or_R} mean',
        f'Cochlear 2 {L_or_R} mean',
        f'Cochlear 3 {L_or_R} mean',
        f'Cochlear 4 {L_or_R} mean',
    ]

    (diffusion_line_spacetime,
     predictions_spacetime,
     u_func) = compare_data_to_model(tac_df, measurement_ROIs, C_dists, D_guess, skipsteps, mouse_id, show_raw_data)

    return (diffusion_line_spacetime, predictions_spacetime, u_func)

def make_predictions(timepoints, distances, u_func, D_guess):

    predictions_spacetime = np.nan * np.zeros((len(timepoints), distances.size))
    for i in tqdm(range(0, predictions_spacetime.shape[0])):
        predictions_spacetime[i, :] = u_func(
            distances,
            timepoints[i],
            D_guess,
        )

    return predictions_spacetime

def compare_data_to_model(
    tac_df,
    measurement_ROIs,
    distances,
    D_guess,
    skipsteps,
    mouse_id='',
    show_raw_data=True,
):

    try:
        diffusion_line_spacetime = tac_df[measurement_ROIs].values
    except KeyError as ke:
        print(ke.args)
        if not "mean'," in str(ke):
            print('Trying with one fewer ROIs')
            diffusion_line_spacetime = tac_df[measurement_ROIs[:-1]].values
            distances = distances[:-1]
        else:
            print('Appears more than 1 ROI is missing, skipping.')
            raise(ke)

    if show_raw_data:
        plot_raw_data_samples(diffusion_line_spacetime, distances, skipsteps)

    t_hat = tac_df.loc[:, 'time'].values
    u_func = create_diff_eq_solution_function(
        h=tac_df.loc[:, measurement_ROIs[0]],
        t=t_hat,
    )

    t_hat = tac_df.loc[::skipsteps, 'time'].values
    predictions_spacetime = make_predictions(
        t_hat,
        distances,
        u_func,
        D_guess,
    )

    plot_data_vs_model(
        tac_df,
        distances,
        diffusion_line_spacetime,
        predictions_spacetime,
        mouse_id,
    )

    if sum(np.isnan(predictions_spacetime.flatten())) != 0:
        print('Failed to fill prediction matrix.')

    return diffusion_line_spacetime, predictions_spacetime, u_func
