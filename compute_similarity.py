import os, sys, librosa
import pandas as pd
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import IPython.display as ipd

from numba import jit

# Computing similarity matrix

@jit(nopython=True)
def compute_basic_similarity_matrix(X:np.array, Y:np.array):
    basic_sim = np.dot(X.T, Y)
    return basic_sim


# Post-processing functions to enhance similarity matrix
@jit(nopython=True)
def shift_cycl_matrix(S, shift):
    """Shifts the matrix similarity to make it correspond to shifted music signals
    in term of musical tonality. 
    Function taken from notebook by Meinard Müller

    Args:
        S (np.ndarray): similarity matrix
        shift (int): 
    Returns:
        shifted_matrix (np.ndarray): Shifted matrix
    """
    N, M = S.shape
    # Handle pooling of value at the matrix's limit
    shift = np.mod(shift, N)
    # Shift matrix 
    shifted_matrix = np.zeros((N, M))
    shifted_matrix[shift:N, :] = S[0:N-shift, :]
    shifted_matrix[0:shift, :] = S[N-shift:N, :]
    return shifted_matrix

@jit(nopython=True)
def filter_diag_mult_sm(S, L=1, tempo_rel_set=np.asarray([1]), direction=0):
    """Path smoothing of similarity matrix by filtering in forward or backward direction
    along various directions around main diagonal.

    Args:
        S (np.ndarray): Self-similarity matrix (SSM)
        L (int): Length of filter (Default value = 1)
        tempo_rel_set (np.ndarray): Set of relative tempo values (Default value = np.asarray([1]))
        direction (int): Direction of smoothing (0: forward; 1: backward) (Default value = 0)

    Returns:
        S_L_final (np.ndarray): Smoothed SM
    """
    N = S.shape[0]
    M = S.shape[1]
    num = len(tempo_rel_set)
    S_L_final = np.zeros((N, M))

    for s in range(0, num):
        M_ceil = int(np.ceil(M / tempo_rel_set[s]))
        resample = np.multiply(np.divide(np.arange(1, M_ceil+1), M_ceil), M)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)
        S_resample = S[:, index_resample]

        S_L = np.zeros((N, M_ceil))
        S_extend_L = np.zeros((N + L, M_ceil + L))

        # Forward direction
        if direction == 0:
            S_extend_L[0:N, 0:M_ceil] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[pos:(N + pos), pos:(M_ceil + pos)]

        # Backward direction
        if direction == 1:
            S_extend_L[L:(N+L), L:(M_ceil+L)] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[(L-pos):(N + L - pos), (L-pos):(M_ceil + L - pos)]

        S_L = S_L / L
        resample = np.multiply(np.divide(np.arange(1, M+1), M), M_ceil)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)

        S_resample_inv = S_L[:, index_resample]
        S_L_final = np.maximum(S_L_final, S_resample_inv)

    return S_L_final


@jit(nopython=True)
def apply_threshold_matrix(S, thresh, strategy='absolute', scale=False, penalty=0.0, binarize=False) :
    """Treshold matrix in a relative fashion

    Args:
        S (np.ndarray): Input matrix
        thresh (float or list): Treshold (meaning depends on strategy)
        strategy (str): Thresholding strategy ('absolute', 'relative', 'local') (Default value = 'absolute')
        scale (bool): If scale=True, then scaling of positive values to range [0,1] (Default value = False)
        penalty (float): Set values below treshold to value specified (Default value = 0.0)
        binarize (bool): Binarizes final matrix (positive: 1; otherwise: 0) (Default value = False)

    Returns:
        threshold_matrix (np.ndarray): Thresholded matrix
    """
    if np.min(S) < 0:
        raise Exception('All entries of the input matrix must be nonnegative')

    threshold_matrix = np.copy(S)
    N, M = S.shape
    num_cells = N * M

    if strategy == 'absolute':
        thresh_abs = thresh
        threshold_matrix[threshold_matrix < thresh] = 0

    if strategy == 'relative':
        thresh_rel = thresh
        num_cells_below_thresh = int(np.round(threshold_matrix.size*(1-thresh_rel)))
        if num_cells_below_thresh < num_cells:
            values_sorted = np.sort(threshold_matrix.flatten('F'))
            thresh_abs = values_sorted[num_cells_below_thresh]
            threshold_matrix[threshold_matrix < thresh_abs] = 0
        else:
            threshold_matrix = np.zeros([N, M])

    if strategy == 'local':
        thresh_rel_row = thresh[0]
        thresh_rel_col = thresh[1]
        S_binary_row = np.zeros([N, M])
        num_cells_row_below_thresh = int(np.round(M * (1-thresh_rel_row)))
        for n in range(N):
            row = S[n, :]
            values_sorted = np.sort(row)
            if num_cells_row_below_thresh < M:
                thresh_abs = values_sorted[num_cells_row_below_thresh]
                S_binary_row[n, :] = (row >= thresh_abs)
        S_binary_col = np.zeros([N, M])
        num_cells_col_below_thresh = int(np.round(N * (1-thresh_rel_col)))
        for m in range(M):
            col = S[:, m]
            values_sorted = np.sort(col)
            if num_cells_col_below_thresh < N:
                thresh_abs = values_sorted[num_cells_col_below_thresh]
                S_binary_col[:, m] = (col >= thresh_abs)
        threshold_matrix = S * S_binary_row * S_binary_col

    if scale:
        cell_val_zero = np.where(threshold_matrix == 0)
        cell_val_pos = np.where(threshold_matrix > 0)
        if len(cell_val_pos[0]) == 0:
            min_value = 0
        else:
            min_value = np.min(threshold_matrix[cell_val_pos])
        max_value = np.max(threshold_matrix)
        # print('min_value = ', min_value, ', max_value = ', max_value)
        if max_value > min_value:
            threshold_matrix = np.divide((threshold_matrix - min_value), (max_value - min_value))
            if len(cell_val_zero[0]) > 0:
                threshold_matrix[cell_val_zero] = penalty
        else:
            print('Condition max_value > min_value is voliated: output zero matrix')

    if binarize:
        threshold_matrix[threshold_matrix > 0] = 1
        threshold_matrix[threshold_matrix < 0] = 0
    return threshold_matrix

