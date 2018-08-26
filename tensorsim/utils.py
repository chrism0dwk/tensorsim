import math
import numpy as np

def quantize(t, Z, intervals=None):
    """
    t @ (t, reps)
    Z @ (t, var, reps)
    """

    if intervals is None:
        n_quantized_t = math.ceil(np.max(t))
        intervals = np.arange(n_quantized_t)
    else:
        n_quantized_t = len(intervals)

    n_t, n_vars, n_reps = Z.shape
    t_q = np.digitize(t.T, intervals)
    rep_idxs, temporal_idxs = np.where(np.roll(t_q, 1, axis=1)!=t_q)
    quantised_idx = t_q[rep_idxs, temporal_idxs]-1
    Z_q = np.zeros((n_reps, n_quantized_t, n_vars))
    Z_q[:] = np.NAN
    Z_q[rep_idxs, quantised_idx, :] = Z[temporal_idxs, :, rep_idxs]

    return intervals, Z_q
