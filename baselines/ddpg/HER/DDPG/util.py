import tensorflow as tf
import numpy as np
import rl_algs.common.tf_util as U
from rl_algs.common.mpi_moments import mpi_moments
import os, subprocess, sys

def flatgrad(loss, var_list, grad_ys=None): # version from rl_algs with an additional argument grad_ys
    grads = tf.gradients(loss, var_list, grad_ys)
    return tf.concat([tf.reshape(grad, [U.numel(v)])
        for (v, grad) in zip(var_list, grads)], 0)

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0][0]

def nn(input, layers_sizes, reuse=None, flatten=False, layer_norm=False):
    for i, size in enumerate(layers_sizes):
        with tf.variable_scope(str(i), reuse=reuse):
            activation = tf.nn.relu if i < len(layers_sizes)-1 else None
            input = tf.layers.dense(inputs=input,
                                    units=size,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    reuse=reuse)
            if layer_norm and i < len(layers_sizes)-1:
                input = tf.contrib.layers.layer_norm(input, scale=True, center=True)
            if activation:
                input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input

def install_mpi_excepthook():
    import sys
    from mpi4py import MPI
    import time
    old_hook = sys.excepthook
    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()
    sys.excepthook = new_hook

def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    install_mpi_excepthook()
    if n<=1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # "-bind-to core" is crucial for good performance
        subprocess.check_call(["mpirun", "-np", str(n), "-bind-to", "core", sys.executable] + sys.argv, env=env)
        return "parent"
    else:
        return "child"