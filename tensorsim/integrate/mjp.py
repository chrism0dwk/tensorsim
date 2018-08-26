"""
Markov Jump Process (mjp)

Continuous time discrete state space model simulation

Implemented algorithms

  * Gillespie direct method (tensorflow)
  * Chemical lagrange equation (con)

@author Leighton Turner
@email leightonturner@gmail.com
"""

import tensorflow as tf
import numpy as np

def mjp(X_t0, theta, S, h_fn, t=None, n_jumps=None):
    """
    Integrate a markov jump process using Gillespie's direct method

    Arguments:

    X_t0    : initial conditions (1, n_X) | (n_reps, n_X)
    theta   : simulation parameters (n_reps, n_parm)
    S       : stoichiometry matrix (n_X, n_rxn)
    h_fn    : hazard function lambda *X, *theta, t: (n_rxn, n_reps)
    t       : time space to integrate over TODO: to be implemented
    n_jumps : integrate for n_jumps


    Returns:
    Y       : integrated timeseries (n_reps, X, n_jumps | n_t)
    """

    # tau = tf.Variable(shape=n_reps ,dtype=tf.float32, name='tau')

    def f_i(S_tm1, _):

        t_tm1, X_tm1 = S_tm1

        parm_ = tf.unstack(X_tm1) + tf.unstack(theta) # X_1 ... X_n, p_1 ... p_k
        p = h_fn(*parm_) #([REP]*RXN)

        # p = h_fn(X_tm1, theta) #([REP]*RXN)

        p_n = sum(p) #(REP) total propensity

        r_u = tf.random_uniform(t_tm1.shape) #(REP)

        tau = tf.multiply((1/p_n),tf.log(1/r_u), name='tau') #(REP)

        rxn = tf.multinomial(tf.transpose(tf.log(p)), 1, name='rxn') #(REP, 1)

        delta = tf.transpose(tf.gather(S, tf.squeeze(rxn)), name='delta') #(X, REP)

        return t_tm1+tau, X_tm1+delta


    n_X, n_reps = X_t0.shape.as_list()
    n_parm, n_reps_,  = theta.shape.as_list()
    assert n_reps == n_reps_  or n_reps_== 1, 'X_t0 and theta must have same leading dimension'

    t0 = tf.zeros(n_reps, name='t0') #(REP, 1)

    intervals = tf.zeros(n_jumps) #(ITERS)

    fn = tf.scan(f_i, intervals, initializer=(t0, X_t0))

    return fn
