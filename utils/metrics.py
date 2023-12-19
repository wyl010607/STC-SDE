import numpy as np


def get_mae(y_pred, y_true):
    non_zero_mask = y_true != 0
    # non_zero_pos = range(y_pred.shape[0])
    return np.fabs((y_true[non_zero_mask] - y_pred[non_zero_mask])).mean()


def get_rmse(y_pred, y_true):
    non_zero_mask = y_true != 0
    # non_zero_pos = range(y_pred.shape[0])
    return np.sqrt(np.square(y_true[non_zero_mask] - y_pred[non_zero_mask]).mean())


def get_mape(y_pred, y_true):
    non_zero_mask = y_true != 0
    y_true_masked = y_true[non_zero_mask]
    y_pred_masked = y_pred[non_zero_mask]
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    return mape


def get_rmspe(y_pred, y_true):
    non_zero_mask = y_true != 0
    rmspe = (
        np.sqrt(
            np.mean(
                np.square(
                    (y_true[non_zero_mask] - y_pred[non_zero_mask])
                    / y_true[non_zero_mask]
                )
            )
        )
        * 100
    )
    return rmspe
