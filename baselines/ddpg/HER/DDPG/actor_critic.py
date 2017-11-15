import tensorflow as tf
from examples.her.normalizer import Normalizer, IdentityNormalizer
from examples.her.util import flatgrad, nn


class ActorCritic:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # placeholders
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.dimx], name="x") # observations
        self.g_tf = tf.placeholder(tf.float32, shape=[None, self.dimg], name="g") # goals
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.dimu], name="u") # actions

        # critic's input
        x = self.x_stats.normalize(self.x_tf)
        g = self.g_stats.normalize(self.g_tf)
        # actor's input
        noisy_x = self.x_tf + tf.random_normal(tf.shape(self.x_tf)) * tf.reshape(self.x_noise_level_tf, [1, self.dimx])
        noisy_g = self.g_tf + tf.random_normal(tf.shape(self.g_tf)) * tf.reshape(self.g_noise_level_tf, [1, self.dimg])
        noisy_x = self.x_stats.normalize(noisy_x)
        noisy_g = self.g_stats.normalize(noisy_g)
        input_pi = tf.concat(axis=1, values=[noisy_x[:,self.pi_input_slice], noisy_g]) # for actor

        # networks
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu], layer_norm=self.layer_norm))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[x, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1], flatten=True, layer_norm=self.layer_norm)
            # for critic training
            input_Q = tf.concat(axis=1, values=[x, g, self.u_tf / self.max_u])
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], flatten=True, reuse=True, layer_norm=self.layer_norm)
