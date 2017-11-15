import numpy as np
from mpi4py import MPI
from rl_algs.common.mpi_moments import mpi_moments
import tensorflow as tf

class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.sum = np.zeros(self.size, np.float32)
        self.sumsq = np.zeros(self.size, np.float32)
        self.count = np.ones(1, np.float32)
        self.mean = tf.Variable(np.zeros((1, self.size), np.float32), name='mean', trainable=False)
        self.std = tf.Variable(np.ones((1, self.size), np.float32), name='std', trainable=False)

        self.mean_new = tf.placeholder(name='mean_new', shape=(1, self.size), dtype=tf.float32)
        self.std_new = tf.placeholder(name='std_new', shape=(1, self.size), dtype=tf.float32)
        self.update_op = tf.group(self.mean.assign(self.mean_new), self.std.assign(self.std_new))

        self.needs_recompute = False

    def update(self, v):
        assert v.ndim == 2
        assert v.shape[1] == self.size
        self.sum += v.sum(axis=0)
        self.sumsq += (np.square(v)).sum(axis=0)
        self.count[0] += v.shape[0]

        self.needs_recompute = True

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return tf.clip_by_value((v - self.mean) / self.std, -clip_range, clip_range)

    def denormalize(self, v):
        return self.mean + v * self.std

    def _mpi_average(self, x):
        buffer = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buffer, op=MPI.SUM)
        buffer /= MPI.COMM_WORLD.Get_size()
        return buffer

    def synchronize(self, root=None):
        self.sum[...] = self._mpi_average(self.sum)
        self.sumsq[...] = self._mpi_average(self.sumsq)
        self.count[...] = self._mpi_average(self.count)

    def recompute_stats(self):
        if not self.needs_recompute:
            return

        self.synchronize()  # TODO: what happens if we are not on MPI?
        mean_new = self.sum / self.count[0]
        std_new = np.sqrt(np.maximum(np.square(self.eps), self.sumsq / self.count[0] - np.square(mean_new)))
        tf.get_default_session().run(self.update_op, feed_dict={
            self.mean_new: mean_new.reshape(1, self.size),
            self.std_new: std_new.reshape(1, self.size),
        })
        self.needs_recompute = False

class IdentityNormalizer:
    def __init__(self, size, std = 1.):
        self.size = size
        self.mean = tf.zeros(self.size, tf.float32)
        self.std = std * tf.ones(self.size, tf.float32)
    def update(self, x):
        pass
    def normalize(self, x, clip_range=None):
        return x/self.std
    def denormalize(self, x):
        return self.std*x
    def synchronize(self):
        pass
    def recompute_stats(self):
        pass
