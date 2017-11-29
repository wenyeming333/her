from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines import bench
import tensorflow as tf
from baselines import logger
from agents import tools
import numpy as np
import time
import gym

def define_batch_env(constructor, num_agents, env_processes):
	"""Create environments and apply all desired wrappers.
	Args:
	constructor: Constructor of an OpenAI gym environment.
	num_agents: Number of environments to combine in the batch.
	env_processes: Whether to step environment in external processes.
	Returns:
	In-graph environments object.
	"""
	with tf.variable_scope('environments'):
		if env_processes:
			envs = [tools.wrappers.ExternalProcess(constructor) 
				for _ in range(num_agents)]
		else:
			envs = [constructor() for _ in range(num_agents)]
		batch_env = tools.BatchEnv(envs, blocking=not env_processes)
		# batch_env = tools.InGraphBatchEnv(batch_env)
	return batch_env

def _create_environment():
	env = gym.make('HindsightReacher-v0')
	return env

# def make_env(rank):
# 	def _thunk():
# 	    env = gym.make('HindsightReacher-v0')
# 	    env.seed(100 + rank)
# 	    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
# 	    # gym.logger.setLevel(logging.WARN)
# 	    return env
# 	return _thunk

# envs = [gym.make('HindsightReacher-v0') for i in range(16)]
# vec_env = SubprocVecEnv([make_env(i) for i in range(16)])

# start = time.time()
# obs = vec_env.reset()
# # for i in range(50):
# actions = np.random.normal([16,4])
# obs, rewards, dones, _ = vec_env.step(actions)
# print('It takes {} secs'.format(time.time()-start))

weights1 = np.random.normal(0,1,[10,64])
weights2 = np.random.normal(0,1,[64,64])
weights3 = np.random.normal(0,1,[64,4])
def network(x):
	x = np.matmul(x,weights1)
	x = np.matmul(x,weights2)
	x = np.matmul(x,weights3)
	return x

batch_env = define_batch_env(lambda: _create_environment(), 
	16, True)

start = time.time()
obs = batch_env.reset()
for i in range(50):
	res = network(obs)
	actions = np.random.normal(0,1,[16,4])
	# obs, rewards, dones, _ = batch_env.step(actions)
print('It takes {} secs'.format(time.time()-start))


env = gym.make('HindsightReacher-v0')
start = time.time()
obs= env.reset()
for j in range(16):
	obs= env.reset()
	for i in range(50):
		res = network(obs)
		# obs, rewards, dones, _ = env.step(np.random.normal([4]))
print('It takes {} secs'.format(time.time()-start))