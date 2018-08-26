import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pl

import tensorsim as ts

""" Model
  dX  = (a*X - B*X*Y)dt  #Prey
  dY  = (B*X*Y - mu*Y)dt #Predator
"""

n_reps = 100
X_t0, Y_t0 = 300., 300.
a, B, mu = 0.1, 0.0005, 0.2

with tf.Graph().as_default():

    # initial conditions
    ics = tf.constant([[X_t0]*n_reps, [Y_t0]*n_reps])

    # simulation parameters
    theta = tf.constant([[a],[B],[mu]])

    #stoichiometric matrix
    S = tf.constant([
        [ 1., 0.],
        [-1., 1.],
        [ 0.,-1.]
    ])

    #hazard function a.k.a propensity vector
    def h_fn(X, Y, a, B, mu):
        return [a*X, B*X*Y, mu*Y]

    Y = ts.integrate.mjp(ics, theta, S, h_fn, n_jumps=30000)

    with tf.Session() as sess:
        # t@(n_jumps, n_reps), Z@(n_jumps, n_var, n_reps)
        t, Z = sess.run(Y)

fig = pl.figure(figsize=(20,12))
for q in range(n_reps):
    pl.plot(t[:,q], Z[:,0,q], "g-", t[:,q], Z[:,1,q], "r-", linewidth=0.05)
pl.ylim((0,1500))
pl.show()
