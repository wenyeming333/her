from gpr import config_tinkerbell
config_tinkerbell()
from tinkerbell import logger
from examples.her.normalizer import Normalizer, IdentityNormalizer
from examples.her.util import flatgrad, nn
from examples.her.actor_critic import ActorCritic
import numpy as np
import tensorflow as tf
import itertools
import math
import warnings
from numpy.linalg import norm
from rl_algs.common.mpi_adam import MpiAdam
import rl_algs.common.tf_util as U
from rl_algs.common.mpi_moments import mpi_moments
from mpi4py import MPI
from gpr.policy import Policy

class DDPG:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if not hasattr(self, 'clip_return'):
            self.clip_return = np.inf

        self.create_network()

        # replay buffer
        self.buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer_x = np.empty((self.buffer_size, self.dimx), np.float32) # state
        self.buffer_u = np.empty((self.buffer_size, self.dimu), np.float32) # action
        self.buffer_gx = np.empty((self.buffer_size, self.dimg), np.float32) # goal (relative to the old state)
        self.buffer_gy = np.empty((self.buffer_size, self.dimg), np.float32) # goal (relative to the new state)
        self.buffer_y = np.empty((self.buffer_size, self.dimx), np.float32) # new state
        self.buffer_r = np.empty((self.buffer_size), np.float32) # reward
        self.buffer_gamma = np.empty(self.buffer_size, np.float32)
        self.current_buffer_size = 0

    def random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_xg(self, x, g):
        x = x.reshape(-1, self.dimx)
        g = g.reshape(-1, self.dimg)
        if self.relative_goals:
            # goals used as inputs are relative to the currently satisfied goal
            # this shouldn't be used if state2goal function is stochastic
            g = g - self.state2goal(x)
        x = np.clip(x, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return x, g

    def pi(self, x, g, noise_eps=0., random_eps=0., param_eps=0., use_target_net=False,
        correlated_noise=None, compute_Q=False):
        x, g = self._preprocess_xg(x, g)
        policy = self.target if use_target_net else self.main
        feed = {
            policy.x_tf : x,
            policy.g_tf : g,
            policy.x_noise_level_tf : self.pi_obs_noise * np.ones((self.dimx,)),
            policy.g_noise_level_tf : self.pi_obs_noise * np.ones((self.dimg,)),
        }
        if param_eps > 0.: #TODO: fix
            if not self.layer_norm:
                warnings.warn('`param_eps > 0` but `layer_norm = False`. If you use param noise, you should normalize using layer norm.')
            actual_pi_tf = self.perturbed_pi_tf

        if compute_Q:
            u, Q = self.sess.run([policy.pi_tf, policy.Q_pi_tf], feed_dict=feed)
        else:
            u = self.sess.run(policy.pi_tf, feed_dict=feed)
            Q = None
        noise = noise_eps * self.max_u * np.random.randn(*u.shape) # gaussian noise
        if correlated_noise is not None:
            noise += correlated_noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1,1) * (self.random_action(u.shape[0]) - u) # eps-greedy

        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()

        ret = (u,)
        if correlated_noise is not None:
            ret += (noise,)
        if compute_Q:
            ret += (Q,)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def adapt_param_noise(self, noise_eps): # TODO: fix
        if self.current_buffer_size == 0:
            return 0., self.param_noise_stddev_tf.eval()
        self.sess.run(self.apply_adaptive_param_noise_tf)
        idx = np.random.randint(0, self.current_buffer_size, size=self.batch_size)
        feed_V = {
            self.x_tf : self.buffer_y[idx],
            self.g_tf : self.buffer_gy[idx],
            self.x_noise_level_tf : self.pi_obs_noise * np.ones((self.dimx,)),
            self.g_noise_level_tf : self.pi_obs_noise * np.ones((self.dimg,)),
        }
        d = self.sess.run(self.adaptive_param_noise_distance_tf, feed_V)
        if d > noise_eps:
            self.sess.run(self.decrease_noise_stddev_tf)
        else:
            self.sess.run(self.increase_noise_stddev_tf)
        return d, self.param_noise_stddev_tf.eval()

    def reset(self):
        pass # FIXME
        #self.sess.run(self.apply_param_noise_tf)

    def sample_buffer_idx(self): # for saving
        if self.current_buffer_size < self.buffer_size:
            idx = range(self.current_buffer_size, self.current_buffer_size + self.rollout_batch_size)
            self.current_buffer_size += self.rollout_batch_size
        else:
            idx = np.random.randint(0, self.buffer_size, size=self.rollout_batch_size)
        return idx

    def store_transitions(self, x, u, y, g, r, gamma):
        x, gx = self._preprocess_xg(x, g)
        y, gy = self._preprocess_xg(y, g)
        for v in [x,u,y,g,r,gamma]:
            assert not np.isnan(v).any()
        idx = self.sample_buffer_idx()
        self.buffer_x[idx] = x
        self.buffer_u[idx] = u
        self.buffer_y[idx] = y
        self.buffer_gx[idx] = gx
        self.buffer_gy[idx] = gy
        self.buffer_r[idx] = r
        self.buffer_gamma[idx] = gamma
        #update running averages
        self.x_stats.update(x)
        self.g_stats.update(gx)

    def sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def grads(self):
        if self.current_buffer_size == 0:
            return
        idx = np.random.randint(0, self.current_buffer_size, size=self.batch_size)
        feed = {
            self.main.x_tf : self.buffer_x[idx],
            self.main.g_tf : self.buffer_gx[idx],
            self.main.u_tf : self.buffer_u[idx],
            self.target.x_tf : self.buffer_y[idx],
            self.target.g_tf : self.buffer_gy[idx],
            self.r_tf : self.buffer_r[idx],
            self.gamma_tf : self.buffer_gamma[idx],
            self.x_noise_level_tf : self.pi_obs_noise * np.ones((self.dimx,)),
            self.g_noise_level_tf : self.pi_obs_noise * np.ones((self.dimg,)),
        }
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([self.Q_loss_tf, self.main.Q_pi_tf, self.Q_grad_tf, self.pi_grad_tf], feed)
        return critic_loss, actor_loss, Q_grad, pi_grad

    def update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def train(self):
        self.x_stats.recompute_stats()
        self.g_stats.recompute_stats()

        critic_loss, actor_loss, Q_grad, pi_grad = self.grads()
        self.update(Q_grad, pi_grad)
        return critic_loss, actor_loss

    def init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.current_buffer_size = 0

    def vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def create_param_noise_subnetwork(self, input_pi):
        with tf.variable_scope('param_noise'):
            with tf.variable_scope('perturbed_pi'):
                self.perturbed_pi_tf = self.max_u * tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu], layer_norm=self.layer_norm))
            with tf.variable_scope('adaptive_pi'):
                self.adaptive_pi_tf = self.max_u * tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu], layer_norm=self.layer_norm))

        # Set up param noise.
        self.param_noise_stddev_tf = tf.Variable(0.1, trainable=False, name='param_noise/stddev')
        param_noise_assigns = []
        assert len(self.global_vars('main/pi')) == len(self.global_vars('param_noise/perturbed_pi'))
        for var, noise_var in zip(self.global_vars('main/pi'), self.global_vars('param_noise/perturbed_pi')):
            if var in self.main_vars and 'LayerNorm' not in var.name:
                # Trainable, apply noise.
                assign = tf.assign(noise_var, var + tf.random_normal(shape=tf.shape(var), mean=0., stddev=self.param_noise_stddev_tf))
            else:
                assign = tf.assign(noise_var, var)
            param_noise_assigns.append(assign)
        assert len(param_noise_assigns) == len(self.global_vars('main/pi'))
        self.apply_param_noise_tf = tf.group(*param_noise_assigns)

        # Set up adaptive adjustments for param noise.
        adaptive_param_noise = []
        assert len(self.global_vars('main/pi')) == len(self.global_vars('param_noise/adaptive_pi'))
        for var, noise_var in zip(self.global_vars('main/pi'), self.global_vars('param_noise/adaptive_pi')):
            if var in self.main_vars and 'LayerNorm' not in var.name:
                # Trainable, apply noise.
                assign = tf.assign(noise_var, var + tf.random_normal(shape=tf.shape(var), mean=0., stddev=self.param_noise_stddev_tf))
            else:
                assign = tf.assign(noise_var, var)
            adaptive_param_noise.append(assign)
        assert len(adaptive_param_noise) == len(self.global_vars('main/pi'))
        self.apply_adaptive_param_noise_tf = tf.group(*adaptive_param_noise)
        self.adaptive_param_noise_distance_tf = tf.sqrt(tf.reduce_mean(tf.square((self.pi_tf - self.adaptive_pi_tf) / self.max_u)))
        self.increase_noise_stddev_tf = tf.assign(self.param_noise_stddev_tf, self.param_noise_stddev_tf * 1.01)
        self.decrease_noise_stddev_tf = tf.assign(self.param_noise_stddev_tf, self.param_noise_stddev_tf / 1.01)

    def create_network(self):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        with tf.variable_scope(self.scope):
            # running averages
            with tf.variable_scope('x_stats'):
                self.x_stats = Normalizer(self.dimx, self.norm_eps, self.norm_clip) if self.normalize_obs else IdentityNormalizer(self.dimx)
            with tf.variable_scope('g_stats'):
                self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip) if self.normalize_obs else IdentityNormalizer(self.dimg)

            # networks
            self.x_noise_level_tf = tf.placeholder(tf.float32, shape=[self.dimx], name="x_noise") # state noise to add for actor
            self.g_noise_level_tf = tf.placeholder(tf.float32, shape=[self.dimg], name="g_noise") # goal noise to add for actor
            with tf.variable_scope('main'):
                self.main = ActorCritic(**self.__dict__)
            with tf.variable_scope('target'):
                self.target = ActorCritic(**self.__dict__)
            assert len(self.vars("main")) == len(self.vars("target"))

            # additional placeholders
            self.r_tf = tf.placeholder(tf.float32, shape=[None], name="r") # reward
            self.gamma_tf = tf.placeholder(tf.float32, shape=[None], name="gamma") # gamma

            # loss functions
            self.target_tf = tf.clip_by_value(self.r_tf + self.gamma_tf * self.target.Q_pi_tf, -self.clip_return, 0. if self.clip_pos_returns else np.inf)
            self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(self.target_tf) - self.main.Q_tf))
            self.Q_grad_tf = flatgrad(self.Q_loss_tf, self.vars('main/Q'))
            self.pi_grad_tf = flatgrad(self.main.pi_tf, self.vars('main/pi'), grad_ys = -tf.gradients(self.main.Q_pi_tf, self.main.pi_tf)[0] + \
                                                                                2. * self.action_l2 * self.main.pi_tf / self.max_u)
            # optimizers
            self.Q_adam = MpiAdam(self.vars('main/Q'), scale_grad_by_procs=False)
            self.pi_adam = MpiAdam(self.vars('main/pi'), scale_grad_by_procs=False)

            # polyak averaging
            self.main_vars = self.vars('main/Q') + self.vars('main/pi')
            self.target_vars = self.vars('target/Q') + self.vars('target/pi')
            self.init_target_net_op = list(map(lambda v : v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
            self.update_target_net_op = list(map(lambda v : v[0].assign(self.polyak * v[0] + (1.-self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

            # Create param space network.
            #self.create_param_noise_subnetwork(input_pi_xg) # TODO: fix

            # initialize all variables
            tf.variables_initializer(self.global_vars('')).run()
            self.sync_optimizers()
            self.init_target_net()
            self.reset()

    def __getstate__(self):
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats', '_tf', 'main', 'target']
        state = {k: v for k, v in self.__dict__.items()
                       if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run(tf.global_variables())
        return state

    def __setstate__(self, state):
        state['scope'] = str(np.random.randint(1000000)) # it makes it possible to load multiple policies trained with the same scope
        state['pi_obs_noise'] = 0. # disable noise
        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/')
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)