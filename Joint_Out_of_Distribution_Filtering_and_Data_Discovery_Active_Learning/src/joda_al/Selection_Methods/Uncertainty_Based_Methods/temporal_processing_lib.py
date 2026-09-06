import numpy as np
from typing import List


def extract_method(method: str, options: List[str]) -> str:
    """

    :param method:
    :param options:
    :return:
    """
    for option in options:
        if option in method:
            return option
    return "plain_time"


def time_processing(method, uncertainty_scores, timestamps, unlabeled_idx_set):
    if method == "Dt":
        set_idx = normal_indices(time_derivative(uncertainty_scores, timestamps, unlabeled_idx_set),unlabeled_idx_set)
    elif method == "Dtp":
        set_idx = plus_indices(time_derivative(uncertainty_scores, timestamps, unlabeled_idx_set), unlabeled_idx_set)
    elif method == "Dtn":
        set_idx = normal_indices(abs_time_derivative(uncertainty_scores, timestamps, unlabeled_idx_set), unlabeled_idx_set)
    elif method == "Dtnp":
        set_idx = plus_indices(abs_time_derivative(uncertainty_scores, timestamps, unlabeled_idx_set), unlabeled_idx_set)
    elif method == "D":
        set_idx = normal_indices(sample_derivative(uncertainty_scores), unlabeled_idx_set)
    elif method == "Dp":
        set_idx = plus_indices(sample_derivative(uncertainty_scores), unlabeled_idx_set)
    elif method == "Dn":
        set_idx = normal_indices(abs_sample_derivative(uncertainty_scores), unlabeled_idx_set)
    elif method == "Dnp":
        set_idx = plus_indices(abs_sample_derivative(uncertainty_scores), unlabeled_idx_set)
    elif method == "plain_time":
        set_idx = normal_indices(plain_uncertainty(uncertainty_scores), unlabeled_idx_set)
    else:
        raise NotImplementedError("Selected Time Processing is not supported")
    return set_idx


def plain_uncertainty(uncertainty_scores):
    indices = np.argsort(uncertainty_scores)[::-1]
    return indices


def time_derivative(uncertainty_scores, timestamps, unlabeled_idx_set):
    """
    d/dt
    :return:
    """
    uncertainty = np.diff(uncertainty_scores)
    dt = np.abs(np.diff(timestamps[unlabeled_idx_set]))
    indices = np.argsort(uncertainty / dt)[::-1]
    return indices


def abs_time_derivative(uncertainty_scores, timestamps, unlabeled_idx_set):
    """
    d/dt
    :return:
    """
    uncertainty = np.abs(np.diff(uncertainty_scores))
    dt = np.abs(np.diff(timestamps[unlabeled_idx_set]))
    indices = np.argsort(uncertainty / dt)[::-1]
    return indices


def sample_derivative(uncertainty_scores):
    """
    d/ds
    :return:
    """
    uncertainty = np.diff(uncertainty_scores)
    indices = np.argsort(uncertainty)[::-1]
    return indices


def abs_sample_derivative(uncertainty_scores):
    """
    d/ds
    Computes the absolute derivative of an array of uncertainty scores.

    Parameters:
    uncertainty_scores (array-like): An array of uncertainty scores.

    :return:
    indices (numpy.ndarray): An array of indices that correspond to the sorted absolute derivative.
    """
    uncertainty = np.abs(np.diff(uncertainty_scores))
    indices = np.argsort(uncertainty)[::-1]
    return indices


def plus_indices(indices, unlabeled_idx_set):
    set_indices = [unlabeled_idx_set[i] + 1 for i in indices]
    return set_indices


def normal_indices(indices, unlabeled_idx_set):
    set_indices = [unlabeled_idx_set[i] for i in indices]
    return set_indices