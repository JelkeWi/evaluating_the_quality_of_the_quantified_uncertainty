import numpy as np
import scipy.stats as stats
import netcal.metrics

def calculate_prediction_interval(means: np.ndarray, stds: np.ndarray, confidence_level=0.95) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the prediction intervals for given confidence levels.
    This is relevant for the PICP and calibration score.

    Parameters:
    -----------
    means : array-like
        Predicted mean values from the model.

    stds : array-like
        Predicted standard deviations from the model.

    confidence_level : float, optional
        The confidence level for the prediction intervals (default is 0.95).

    Returns:
    --------
    tuple
        A tuple containing two arrays:
        - lower_bounds : Lower bounds of the prediction intervals.
        - upper_bounds : Upper bounds of the prediction intervals.
    """
    # Find the Z-score corresponding to the confidence level - calculates inverse CDF
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate the margin of error using the standard deviation of the means
    margin_of_error = z_score * stds

    # Compute the confidence interval for all predictions
    lower_bounds = np.squeeze(means - margin_of_error)
    upper_bounds = np.squeeze(means + margin_of_error)

    return lower_bounds, upper_bounds



# Kuleshov
def calculate_calibration_score(mean_values: np.ndarray, std_values: np.ndarray, true_values: np.ndarray, confidence_levels: list) -> tuple[list, float]:
    """
    Computes the calibration error based on predicted means, standard deviations, and true values.

    Parameters:
    -----------
    mean_values : array-like
        Predicted mean values from the model.

    std_values : array-like
        Predicted standard deviations from the model.

    true_values : array-like
        True values to compare against the predictions.

    confidence_levels : list of float
        List of confidence levels for which to compute the calibration error.

    Returns:
    --------
    tuple
        A tuple where the first element is a list of empirical frequencies for each confidence level,
        and the second element is the average calibration error.
    """
    empirical_frequencies = []

    # Calculate empirical frequencies for each confidence level
    for confidence_level in confidence_levels:
        empirical_frequencies.append(netcal.metrics.PICP().measure(X=(mean_values, std_values), y=true_values, q=confidence_level).picp.squeeze())
                                
    # Calculate calibration error
    calibration_error = 0.0
    for confidence_level, empirical_frequency_pi in zip(confidence_levels, empirical_frequencies):
        # Calculate the weighted squared difference
        weight = 1.0  # standard
        error = (confidence_level - empirical_frequency_pi) ** 2
        weighted_error = weight * error

        calibration_error += weighted_error

    # Normalize calibration error
    calibration_error /= len(confidence_levels)

    return empirical_frequencies, calibration_error


def calculate_coverage_width_based_criterion(lower_bounds_pi: np.ndarray, upper_bounds_pi: np.ndarray, picp: float, target_range: float, confidence_level: float = 0.95, eta: float = 50.0) -> float:
    """Calculates the coverage width-based criterion (CWC).

    Args:
        lower_bounds_pi (np.ndarray): Lower bounds of the PIs
        upper_bounds_pi (np.ndarray): Upper bound of the PIs
        picp (float): Pridiction interval coverage probability
        target_range (float): Range of the target value (for normalization of the PIs)
        confidence_level (float, optional): Confidence level of the PICP. Defaults to 0.95.
        eta (float, optional): Scaling hyperparameter of the CWC. Defaults to 50.0.

    Returns:
        float: CWC
    """
    mpiw = np.mean(upper_bounds_pi - lower_bounds_pi)

    nmpiw = mpiw / target_range
    if picp >= confidence_level:
        cwc = nmpiw
    else:
        cwc = nmpiw * (1 + np.exp(-eta * (picp - confidence_level)))
    return cwc


def calculate_expectation_coverage_probability_error(mean_values: np.ndarray, std_values: np.ndarray, true_values: np.ndarray, confidence_levels: list) -> float:
    """
    Computes the expectation coverage probability error (ECPE) based on predicted means, standard deviations, and true values.

    Parameters:
    -----------
    mean_values : array-like
        Predicted mean values from the model.

    std_values : array-like
        Predicted standard deviations from the model.

    true_values : array-like
        True values to compare against the predictions.

    confidence_levels : list of float
        List of confidence levels for which to compute the calibration error.

    Returns:
    --------
    tuple
        A tuple where the first element is a list of empirical frequencies for each confidence level,
        and the second element is the average calibration error.
    """
    picps = np.zeros_like(confidence_levels)

    # Calculate picp for each confidence level
    for i, confidence_level in enumerate(confidence_levels):
        picps[i] = netcal.metrics.PICP().measure(X=(mean_values, std_values), y=true_values, q=confidence_level).picp.squeeze()

    ecpe = np.mean(np.abs(confidence_levels - picps))

    return ecpe
