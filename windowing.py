import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple

def create_rolling_windows(data: np.ndarray, history_len: int, horizon_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates rolling windows from time series data, as described in the paper "Do we really need deep learning for timeseries forecasting?".

    Args:
        data: numpy array of shape (num_timeseries, num_observations, num_features), where the first feature corresponds to the target.
        history_len: integer specifying the length of the target history to use for each window.
        horizon_len: integer specifying the length of the target horizon to use for each window.

    Returns:
        A tuple containing:
        - X: numpy array of shape (num_windows, history_len + num_features - 1), containing the input data for each window.
        - y: numpy array of shape (num_windows, horizon_len), containing the target data for each window.
    """
    window_size = history_len + horizon_len
    num_features = data.shape[-1]

    # sliding_window_view returns a windowed view into the data with shape (after squeeze) of
    # (num_timeseries, num_windows, window_size, num_features)
    windows = sliding_window_view(data, (window_size, num_features), (1,2), writeable=False).squeeze(axis=-3)
    
    # up until last observation, we only use the history of the target (and disregard covariates):
    X = windows[:,: ,:history_len -1,0]
    
    # only for the last observation of the history, we also concatenate the covariates to our input:
    last_obs_X_plus_covariates = windows[:,: ,history_len -1,:]
    X = np.concatenate((X,last_obs_X_plus_covariates), axis=2)
    
    y = windows[:,:,history_len:,0]

    # flatten into instances
    X = X.reshape(-1, X.shape[-1])
    y = y.reshape(-1, y.shape[-1])

    return X, y