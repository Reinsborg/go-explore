#
# OBS
# This file is not part of the original release by Uber Technologies, Inc.
# Alteration made by:
# Jeppe Reinsborg, 3 June 2019
#


import tensorflow as tf
import numpy as np
import gym
from mlsh_code import rollouts
from mlsh_code.learner import Learner

from goexplore_py import ppo2, policies
from baselines.common.atari_wrappers import *
from mpi4py import MPI
import rl_algs.common.tf_util as U

def clipreward(newcell, gameReward, grid, seen):
	return ((newcell not in grid) and newcell not in seen) + np.clip(gameReward,-1,1)
def IRonly(newcell, gameReward, grid, seen):
	return (newcell not in grid) and (newcell not in seen)


class strechedObSpaceWrapper(gym.ObservationWrapper):
	def __init__(self, env):
		gym.ObservationWrapper.__init__(self, env)
		self.n =env.unwrapped.n
		self.observation_space = spaces.MultiBinary(self.n)

	def observation(self, observation):
		res = np.zeros(self.n)
		res[observation] = 1
		return res

class MyEpisodicLifeEnv(gym.Wrapper):
	def __init__(self, env):
		"""Make end-of-life == end-of-episode, and reset as normal
		"""
		gym.Wrapper.__init__(self, env)
		self.lives = 0
		self.was_real_done  = True

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self.was_real_done = done
		# check current lives, make loss of life terminal,
		# then update lives to handle bonus lives
		lives = self.env.unwrapped.ale.lives()
		if lives < self.lives:
			# for Qbert sometimes we stay in lives == 0 condtion for a few frames
			# so its important to keep lives > 0, so that we only reset once
			# the environment advertises done.
			done = True
		self.lives = lives
		return obs, reward, done, info

	def reset(self, **kwargs):
		return  self.env.reset(**kwargs)

class PPOExplorer:
	def __init__(self, env,  nexp, lr, lr_decay=1, cl_decay=1, nminibatches=4, n_tr_epochs=4, cliprange=0.1, gamma=0.99, lam=0.95, nenvs=1, policy=policies.CnnPolicy):
		ob_space = env.observation_space
		ac_space = env.action_space

		self.exp = 0
		self.nsteps = nexp
		self.batch = self.nsteps * nenvs
		self.n_mb = nminibatches
		self.nbatch_train = self.batch // self.n_mb
		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		self.lr = lr
		self.lr_decay = lr_decay
		self.cl_decay = cl_decay
		self.states = None
		self.done = [False for _ in range(nenvs)]
		self.gamma = gamma
		self.lam = lam
		self.nenvs = nenvs


		self.n_train_epoch = n_tr_epochs
		self.cliprange = cliprange
		self.model = ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
								nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=0.01, vf_coef=1, max_grad_norm=0.5)

		self.obs = np.zeros((nenvs,) + ob_space.shape, dtype=self.model.train_model.X.dtype.name)


	def init_seed(self):
		# self.exp = 0
		# self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		pass

	def init_trajectory(self, arg=None, arg2=None):
		pass

	def seen_state(self, e):
		self.exp += 1
		self.obs[:] = e.obs
		self.mb_rewards.append(e.reward)
		self.done = e.done

		if self.exp >= self.nsteps:
			self.mb_obs = np.asarray(self.mb_obs, dtype=self.obs.dtype).squeeze()
			self.mb_rewards = np.asarray(self.mb_rewards, dtype=np.float32).squeeze()
			self.mb_actions = np.asarray(self.mb_actions).squeeze()
			self.mb_values = np.asarray(self.mb_values, dtype=np.float32).squeeze()
			self.mb_neglogpacs = np.asarray(self.mb_neglogpacs, dtype=np.float32).squeeze()
			self.mb_dones = np.asarray(self.mb_dones, dtype=np.bool).squeeze()



			# From baselines' ppo2 runner():
			# Calculate returns

			last_values = self.model.value(self.obs)

			mb_returns = np.zeros_like(self.mb_rewards)
			mb_advs = np.zeros_like(self.mb_rewards)
			lastgaelam = 0
			for t in reversed(range(self.nsteps)):
				if t == self.nsteps - 1:
					nextnonterminal = 1.0 - self.done
					nextvalues = last_values
				else:
					nextnonterminal = 1.0 - self.mb_dones[t + 1]
					nextvalues = self.mb_values[t + 1]
				delta = self.mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.mb_values[t]
				mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

			mb_returns[:] = mb_advs + self.mb_values
			# Swap and flatten axis 0 and 1
			if self.nenvs > 1:
				self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs = \
					map(ppo2.sf01, (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))


			#train model for multiple epoch in n minibacthes pr epoch
			#From baselines' ppo2 learn()
			inds = np.arange(self.batch)
			for _ in range(self.n_train_epoch):
				np.random.shuffle(inds)
				for start in range(0, self.batch, self.nbatch_train):
					end = start + self.nbatch_train
					mbinds = inds[start:end]
					slices = (arr[mbinds] for arr in (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))
					self.model.train(self.lr, self.cliprange, *slices)





			self.lr *= self.lr_decay
			self.cliprange *=  self.cl_decay
			self.exp = 0
			self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [], [], [], [], [], []

	def get_action(self, state, env):

		self.obs[:] = state

		actions, values, self.states, neglogpacs = self.model.step(self.obs)

		self.mb_obs.append(self.obs.copy())
		self.mb_actions.append(actions)
		self.mb_values.append(values)
		self.mb_neglogpacs.append(neglogpacs)
		self.mb_dones.append(self.done)

		return actions

class PPOExplorer_v2:
	def __init__(self, actors,  nexp, lr, lr_decay=1, cl_decay=1, nminibatches=4, n_tr_epochs=4, cliprange=0.1, gamma=0.99, lam=0.95):


		self.nacts = actors
		self.actor = 0

		self.exp = 0
		self.nsteps = nexp
		self.batch = self.nsteps * actors
		self.n_mb = nminibatches
		self.nbatch_train = self.batch // self.n_mb
		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]],[[]],[[]],[[]],[[]],[[]]
		self.lr = lr
		self.lr_decay = lr_decay
		self.cl_decay = cl_decay
		self.states = None
		self.done = [False for _ in range(1)]
		self.gamma = gamma
		self.lam = lam



		self.n_train_epoch = n_tr_epochs
		self.cliprange = cliprange
		self.model = None

		self.obs = None

	def init_model(self, ob_space, ac_space, policy=policies.CnnPolicy):

		self.model = ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
								nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=0.01, vf_coef=1,
								max_grad_norm=0.5)
		self.obs = np.zeros((1,) + ob_space.shape, dtype=self.model.train_model.X.dtype.name)

	def init_seed(self):
		# self.exp = 0
		# self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		pass

	def init_trajectory(self, arg=None, arg2=None):

		pass

	def seen_state(self, e):
		self.exp += 1
		self.obs[:] = e.obs
		self.mb_rewards[self.actor].append(e.reward)
		self.done = e.done

		if self.exp >= self.nsteps:
			self.actor += 1
			self.exp = 0



			if self.actor != self.nacts:
				self.mb_obs.append([])
				self.mb_actions.append([])
				self.mb_values.append([])
				self.mb_neglogpacs.append([])
				self.mb_dones.append([])
				self.mb_rewards.append([])
			else:
				self.actor = 0

				self.mb_obs = np.asarray(self.mb_obs, dtype=self.obs.dtype,).squeeze()
				self.mb_rewards = np.asarray(self.mb_rewards, dtype=np.float32).squeeze()
				self.mb_actions = np.asarray(self.mb_actions).squeeze()
				self.mb_values = np.asarray(self.mb_values, dtype=np.float32).squeeze()
				self.mb_neglogpacs = np.asarray(self.mb_neglogpacs, dtype=np.float32).squeeze()
				self.mb_dones = np.asarray(self.mb_dones, dtype=np.bool).squeeze()

				self.mb_obs = self.mb_obs.swapaxes(0, 1)
				self.mb_rewards = self.mb_rewards.swapaxes(0, 1)
				self.mb_actions = self. mb_actions.swapaxes(0, 1)
				self.mb_values = self.mb_values.swapaxes(0, 1)
				self.mb_neglogpacs = self.mb_neglogpacs.swapaxes(0, 1)
				self.mb_dones = self.mb_dones.swapaxes(0, 1)

				# From baselines' ppo2 runner():
				# Calculate returns

				last_values = self.model.value(self.obs)

				mb_returns = np.zeros_like(self.mb_rewards)
				mb_advs = np.zeros_like(self.mb_rewards)
				lastgaelam = 0
				for t in reversed(range(self.nsteps)):
					if t == self.nsteps - 1:
						nextnonterminal = 1.0 - self.done
						nextvalues = last_values
					else:
						nextnonterminal = 1.0 - self.mb_dones[t + 1]
						nextvalues = self.mb_values[t + 1]
					delta = self.mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.mb_values[t]
					mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

				mb_returns[:] = mb_advs + self.mb_values
				# Swap and flatten axis 0 and 1
				if self.nacts > 1:
					self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs = \
						map(ppo2.sf01, (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))


				#train model for multiple epoch in n minibacthes pr epoch
				#From baselines' ppo2 learn()
				inds = np.arange(self.batch)
				for _ in range(self.n_train_epoch):
					np.random.shuffle(inds)
					for start in range(0, self.batch, self.nbatch_train):
						end = start + self.nbatch_train
						mbinds = inds[start:end]
						slices = (arr[mbinds] for arr in (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))
						self.model.train(self.lr, self.cliprange, *slices)





				self.lr *= self.lr_decay
				self.cliprange *=  self.cl_decay

				self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]], [[]], [[]], [[]], [[]], [[]]

	def get_action(self, state, env):

		self.obs[:] = env.obs

		actions, values, self.states, neglogpacs = self.model.step(self.obs)

		self.mb_obs[self.actor].append(self.obs.copy())
		self.mb_actions[self.actor].append(actions)
		self.mb_values[self.actor].append(values)
		self.mb_neglogpacs[self.actor].append(neglogpacs)
		self.mb_dones[self.actor].append(self.done)

		return actions

	def __repr__(self):
		return 'ppo'

class PPOExplorer_v3:
	def __init__(self, actors,  nexp, lr, lr_decay=1, cl_decay=1, nminibatches=4, n_tr_epochs=4, cliprange=0.1, gamma=0.99, lam=0.95, name='model', nframes=4, ent_coef=0.01):


		self.nacts = actors
		self.actor = 0

		self.exp = 0
		self.nsteps = nexp
		self.batch = self.nsteps * actors
		self.n_mb = nminibatches
		self.nbatch_train = self.batch // self.n_mb
		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]],[[]],[[]],[[]],[[]],[[]]
		self.lr = lr
		self.lr_decay = lr_decay
		self.cl_decay = cl_decay
		self.states = None
		self.done = [False for _ in range(1)]
		self.gamma = gamma
		self.lam = lam
		self.ent_coef = ent_coef
		self.nframes = nframes



		self.n_train_epoch = n_tr_epochs
		self.cliprange = cliprange
		self.model = None

		self.obs = None
		#self.env = None

		self.name = name

	def init_model(self, env, policy=policies.CnnPolicy):
		# self.env = gym.make(env)
		# if self.env.__repr__() != '<TimeLimit<NChainEnv<NChain-v0>>>':
		# 	self.env = ClipRewardEnv(FrameStack(WarpFrame(self.env), 4))
		# else:
		# 	self.env = self.env.unwrapped
		# 	self.env.unwrapped.n = 10000  #if nchain environment set N to 10 000
		# 	self.env = strechedObSpaceWrapper(self.env)
		# 	#TODO Should not be hardcoded
		# 	self.env.unwrapped.slip = 0


		ob_space = env.observation_space
		ac_space = env.action_space
		self.model = ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
								nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=self.ent_coef, vf_coef=1,
								max_grad_norm=0.5, name=self.name)
		self.obs = np.zeros((1,) + ob_space.shape, dtype=self.model.train_model.X.dtype.name)

	def init_seed(self):
		# self.exp = 0
		# self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		pass

	def init_trajectory(self, obs, arg2):

		# if start_cell.restore is not None:
		# 	if self.env.unwrapped.spec._env_name != "NChain":
		# 		(full_state, state, score, steps, pos, room_time, ram_death_state,_, _) = start_cell.restore
		# 		self.env.unwrapped.restore_full_state(full_state)
		# 		for i in range(3): #TODO this puts the env out of sync
		# 			self.env.step(0) #perform 3(4) nop to fill FrameStack
		# 	else:
		# 		state, _, _, _ = start_cell.restore
		# 		self.env.unwrapped.state = state
		# 	self.obs[:], _ , self.done, _ = self.env.step(0)
		# else:
		# 	self.obs[:] = self.env.reset()
		# 	self.done = [False]
		self.obs[:] = obs
		self.done = True # Delayed done flag to seperate from the previous episode which may not have ended in death

	def seen_state(self, e):

		self.exp += 1

		#self.obs[:], reward, self.done, _ = self.env.step(self.mb_actions[self.actor][-1].squeeze())

		self.obs[:] = e['observation']
		self.mb_rewards[self.actor].append(e['reward'])
		self.done = e['done']

		if self.exp >= self.nsteps:
			self.actor += 1
			self.exp = 0



			if self.actor != self.nacts:
				self.mb_obs.append([])
				self.mb_actions.append([])
				self.mb_values.append([])
				self.mb_neglogpacs.append([])
				self.mb_dones.append([])
				self.mb_rewards.append([])
			else:
				self.actor = 0

				self.mb_obs = np.asarray(self.mb_obs, dtype=self.obs.dtype,).squeeze()
				self.mb_rewards = np.asarray(self.mb_rewards, dtype=np.float32).squeeze()
				self.mb_actions = np.asarray(self.mb_actions).squeeze()
				self.mb_values = np.asarray(self.mb_values, dtype=np.float32).squeeze()
				self.mb_neglogpacs = np.asarray(self.mb_neglogpacs, dtype=np.float32).squeeze()
				self.mb_dones = np.asarray(self.mb_dones, dtype=np.bool).squeeze()

				if self.nacts > 1:
					self.mb_obs = self.mb_obs.swapaxes(0, 1)
					self.mb_rewards = self.mb_rewards.swapaxes(0, 1)
					self.mb_actions = self. mb_actions.swapaxes(0, 1)
					self.mb_values = self.mb_values.swapaxes(0, 1)
					self.mb_neglogpacs = self.mb_neglogpacs.swapaxes(0, 1)
					self.mb_dones = self.mb_dones.swapaxes(0, 1)

				# From baselines' ppo2 runner():
				# Calculate returns

				last_values = self.model.value(self.obs)

				mb_returns = np.zeros_like(self.mb_rewards)
				mb_advs = np.zeros_like(self.mb_rewards)
				lastgaelam = 0
				for t in reversed(range(self.nsteps)):
					if t == self.nsteps - 1:
						nextnonterminal = 1.0 - self.done
						nextvalues = last_values
					else:
						nextnonterminal = 1.0 - self.mb_dones[t + 1]
						nextvalues = self.mb_values[t + 1]
					delta = self.mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.mb_values[t]
					mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

				mb_returns[:] = mb_advs + self.mb_values
				# Swap and flatten axis 0 and 1
				if self.nacts > 1:
					self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs = \
						map(ppo2.sf01, (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))


				#train model for multiple epoch in n minibacthes pr epoch
				#From baselines' ppo2 learn()
				inds = np.arange(self.batch)
				for _ in range(self.n_train_epoch):
					np.random.shuffle(inds)
					for start in range(0, self.batch, self.nbatch_train):
						end = start + self.nbatch_train
						mbinds = inds[start:end]
						slices = (arr[mbinds] for arr in (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))
						self.model.train(self.lr, self.cliprange, *slices)





				self.lr *= self.lr_decay
				self.cliprange *=  self.cl_decay

				self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]], [[]], [[]], [[]], [[]], [[]]

	def get_action(self, state, env):

		#self.obs[:] = env.obs

		actions, values, self.states, neglogpacs = self.model.step(self.obs)

		self.mb_obs[self.actor].append(self.obs.copy())
		self.mb_actions[self.actor].append(actions)
		self.mb_values[self.actor].append(values)
		self.mb_neglogpacs[self.actor].append(neglogpacs)
		self.mb_dones[self.actor].append(self.done)



		return actions

	def clear(self):
		self.model.clear()

	def __repr__(self):
		return 'ppo'


class MlshExplorer:
	def __init__(self, nsubs, timedialation, warmup_T, train_T,  actors, nexp, lr_mas, lr_sub, retrain_N = None,
				 lr_decay=1., cl_decay=1., lr_decay_sub=1., cl_decay_sub=1., nminibatches=4, n_tr_epochs=4,
				 cliprange_mas=0.1, cliprange_sub = 0.1, gamma=0.99, lam=0.95, ent_m=0.01, ent_s=0.01):





		# PPO related
		self.nacts = actors
		self.actor = 0
		self.exp = 0
		self.nsteps = nexp
		self.batch = self.nsteps * actors
		self.n_mb = nminibatches
		self.nbatch_train = self.batch // self.n_mb
		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs, \
			self.mb_domains = [[]],[[]],[[]],[[]],[[]], [[]], [[]]
		self.lr = lr_mas
		self.lr_decay = lr_decay
		self.cl_decay = cl_decay
		self.master_lr = lr_mas
		self.master_cl = cliprange_mas
		self.states = None
		self.done = [False for _ in range(1)]
		self.gamma = gamma
		self.lam = lam
		self.ent = ent_m
		self.n_train_epoch = n_tr_epochs
		self.cliprange = cliprange_mas

		# MLSH
		self.master = None
		self.subs = [PPOSub(actors,  nexp*timedialation, lr_sub, lr_decay_sub, cl_decay_sub, nminibatches, n_tr_epochs, cliprange_sub,
									gamma, lam, name=f'Sub_{i}', ent=ent_s) for i in range(nsubs)]
		self.nsubs = nsubs
		self.time_dialation = timedialation
		self.warm_up_T = warmup_T
		self.warm_up_done = False
		self.train_t = train_T
		self.retrain_N = retrain_N
		self.cur_sub = None
		self.t = 0
		self.reward = 0
		self.reset_count = 0


		self.obs = None
		self.domain = []
		self.env = None

	def init_model(self, env, domain_shape=None, masterPolicy=policies.CnnPolicy, subPolicies=policies.CnnPolicy):
		# self.env = gym.make(env)
		# if self.env.__repr__() != '<TimeLimit<NChainEnv<NChain-v0>>>':
		# 	self.env = ClipRewardEnv(FrameStack(WarpFrame(self.env), 4))
		# else:
		# 	self.env = self.env.unwrapped
		# 	self.env.unwrapped.n = 10000  #if nchain environment set N to 10 000
		# 	self.env = strechedObSpaceWrapper(self.env)
		# 	#TODO Should not be hardcoded
		# 	self.env.unwrapped.slip = 0

		ob_space = env.observation_space
		ac_space = gym.spaces.Discrete(len(self.subs))
		if  masterPolicy == policies.CnnPolicy_withDomain: #isinstance(masterPolicy., policies.CnnPolicy_withDomain):
			assert domain_shape is not None, Exception('domain policy but no domain shape suplied')
			self.master = ppo2.Model(policy=masterPolicy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
									 nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=self.ent, vf_coef=1,
									 max_grad_norm=0.5, name='Master', domain_shape=domain_shape)
			self.domain = np.zeros((1,) + domain_shape, dtype=self.master.train_model.G.dtype.name)
		else:
			self.master = ppo2.Model(policy=masterPolicy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
								nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=self.ent, vf_coef=1,
								max_grad_norm=0.5, name='Master')
		for sub in self.subs:
			sub.init_model(ob_space=ob_space, ac_space=env.action_space, policy=subPolicies)
		# self.subs = [ppo2.Model(policy=masterPolicy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
		# 						nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=0.01, vf_coef=1,
		# 						max_grad_norm=0.5, name=f'Sub_{i}') for i in range(self.nsubs)]
		self.obs = np.zeros((1,) + ob_space.shape, dtype=self.master.train_model.X.dtype.name)


	def init_seed(self):
		# self.exp = 0
		# self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		pass

	def init_trajectory(self, obs, domain):

		# if start_cell.restore is not None:
		# 	if self.env.unwrapped.spec._env_name != "NChain":
		# 		(full_state, state, score, steps, pos, room_time, ram_death_state,_, _) = start_cell.restore
		# 		self.env.unwrapped.restore_full_state(full_state)
		# 		for i in range(3): #TODO this puts the env out of sync
		# 			self.env.step(0) #perform 3(4) nop to fill FrameStack
		# 	else:
		# 		state, _, _, _ = start_cell.restore
		# 		self.env.unwrapped.state = state
		# 	self.obs[:], _ , self.done, _ = self.env.step(0)
		# else:
		# 	self.obs[:] = self.env.reset()
		# 	self.done = [False]
		self.obs[:] = obs
		self.domain[:] = domain
		self.done = True # Delayed done flag to seperate from the previous episode which may not have ended in death
		if self.t % self.time_dialation:
			self.t += (self.time_dialation - (self.t % self.time_dialation)) # fast-forward to next master time step
			#save partial exp
			self.mb_rewards[self.actor].append(self.reward)
			self.reward = 0
			self.exp += 1

			# if this complete a batch train before continuing
			if self.exp >= self.nsteps:
				self.actor += 1
				self.exp = 0

				if self.actor != self.nacts:
					self.mb_obs.append([])
					self.mb_actions.append([])
					self.mb_values.append([])
					self.mb_neglogpacs.append([])
					self.mb_dones.append([])
					self.mb_rewards.append([])
					self.mb_domains.append([])
				else:
					self.actor = 0
					self.train()



	def seen_state(self, e):


		self.t += 1
		#self.obs[:], reward, self.done, _ = self.env.step(self.mb_actions[self.actor][-1].squeeze())

		self.obs[:] = e['observation']
		self.domain[:] = e['domain']
		self.done = e['done']
		self.reward += e['reward']



		if self.warm_up_done:
			self.subs[self.cur_sub].seen_state(e)







		if self.t % self.time_dialation == 0:
			if not self.warm_up_done and self.t >= self.warm_up_T:
				self.warm_up_done = True
				self.t = 0
			self.mb_rewards[self.actor].append(self.reward)
			self.reward = 0
			self.exp += 1

		if self.warm_up_done:

			if self.t >= self.train_t:

				if self.retrain_N is not None and self.reset_count >= self.retrain_N:
					self.t = 0 # assuming subPolicies have convergede enough that no further resets are needed
				else:
					self.reset_master()
					self.warm_up_done = False # reset master policy and reenter warmup period


		if self.exp >= self.nsteps:
			self.actor += 1
			self.exp = 0



			if self.actor != self.nacts:
				self.mb_obs.append([])
				self.mb_actions.append([])
				self.mb_values.append([])
				self.mb_neglogpacs.append([])
				self.mb_dones.append([])
				self.mb_rewards.append([])
				self.mb_domains.append([])
			else:
				self.actor = 0
				self.train()

	def train(self):
		self.mb_obs = np.asarray(self.mb_obs, dtype=self.obs.dtype).squeeze()
		self.mb_rewards = np.asarray(self.mb_rewards, dtype=np.float32).squeeze()
		self.mb_actions = np.asarray(self.mb_actions).squeeze()
		self.mb_values = np.asarray(self.mb_values, dtype=np.float32).squeeze()
		self.mb_neglogpacs = np.asarray(self.mb_neglogpacs, dtype=np.float32).squeeze()
		self.mb_dones = np.asarray(self.mb_dones, dtype=np.bool).squeeze()
		if isinstance(self.master.train_model, policies.CnnPolicy_withDomain):
			self.mb_domains = np.asarray(self.mb_domains, dtype=self.domain.dtype).squeeze(axis=(0,2))
		else:
			self.mb_domains = np.asarray(self.mb_domains).squeeze(axis=0)

		if self.nacts > 1:
			self.mb_obs = self.mb_obs.swapaxes(0, 1)
			self.mb_rewards = self.mb_rewards.swapaxes(0, 1)
			self.mb_actions = self.mb_actions.swapaxes(0, 1)
			self.mb_values = self.mb_values.swapaxes(0, 1)
			self.mb_neglogpacs = self.mb_neglogpacs.swapaxes(0, 1)
			self.mb_dones = self.mb_dones.swapaxes(0, 1)
			if isinstance(self.master.train_model, policies.CnnPolicy_withDomain):
				self.mb_domains = self.mb_domains.swapaxes(0, 1)

		# From baselines' ppo2 runner():
		# Calculate returns

		last_values = self.master.value(self.obs, self.domain)

		mb_returns = np.zeros_like(self.mb_rewards)
		mb_advs = np.zeros_like(self.mb_rewards)
		lastgaelam = 0
		for t in reversed(range(self.nsteps)):
			if t == self.nsteps - 1:
				nextnonterminal = 1.0 - self.done
				nextvalues = last_values
			else:
				nextnonterminal = 1.0 - self.mb_dones[t + 1]
				nextvalues = self.mb_values[t + 1]
			delta = self.mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.mb_values[t]
			mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

		mb_returns[:] = mb_advs + self.mb_values
		# Swap and flatten axis 0 and 1
		if self.nacts > 1:
			self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs, self.mb_domains = \
				map(ppo2.sf01, (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs, self.mb_domains))


		#train model for multiple epoch in n minibacthes pr epoch
		#From baselines' ppo2 learn()
		inds = np.arange(self.batch)
		for _ in range(self.n_train_epoch):
			np.random.shuffle(inds)
			for start in range(0, self.batch, self.nbatch_train):
				end = start + self.nbatch_train
				mbinds = inds[start:end]
				slices = (arr[mbinds] for arr in (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs, self.mb_domains))
				slices = dict(zip(['obs', 'returns', 'masks', 'actions', 'values', 'neglogpacs', 'domains'], slices))
				if not isinstance(self.master.train_model, policies.CnnPolicy_withDomain):
					slices['domains'] = None
				self.master.train(self.lr, self.cliprange, **slices)





		self.lr *= self.lr_decay
		self.cliprange *=  self.cl_decay

		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs, \
		self.mb_domains = [[]], [[]], [[]], [[]], [[]], [[]], [[]]

	def get_action(self, state, env):

		#self.obs[:] = env.obs

		if self.t % self.time_dialation == 0:
			self.cur_sub, master_values, _, master_neglogpacs = self.master.step(self.obs, self.domain)
			self.cur_sub = self.cur_sub.squeeze()
			self.subs[self.cur_sub].init_trajectory(self.obs, None)

			self.mb_obs[self.actor].append(self.obs.copy())
			self.mb_domains[self.actor].append((self.domain.copy()))
			self.mb_actions[self.actor].append(self.cur_sub)
			self.mb_values[self.actor].append(master_values)
			self.mb_neglogpacs[self.actor].append(master_neglogpacs)
			self.mb_dones[self.actor].append(self.done)

		actions = self.subs[self.cur_sub].get_action(self.obs, self.warm_up_done)

		return actions

	def reset_master(self):
		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]], [[]], [[]], [[]], [[]], [[]]
		self.exp = 0
		self.t = 0
		self.actor = 0
		self.cliprange = self.master_cl
		self.lr = self.master_lr
		self.master.reset()



	def __repr__(self):
		return 'mlsh'


class PPOSub:
	def __init__(self, actors, nexp, lr, lr_decay=1., cl_decay=1., nminibatches=4, n_tr_epochs=4, cliprange=0.1,
				 gamma=0.99, lam=0.95, name='model', ent=0.01):

		self.nacts = actors
		self.actor = 0

		self.exp = 0
		self.nsteps = nexp
		self.batch = self.nsteps * actors
		self.n_mb = nminibatches
		self.nbatch_train = self.batch // self.n_mb
		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]], [[]], [[]], [[]], [[]], [[]]
		self.lr = lr
		self.lr_decay = lr_decay
		self.cl_decay = cl_decay
		self.states = None
		self.done = [False for _ in range(1)]
		self.gamma = gamma
		self.lam = lam
		self.ent = ent

		self.n_train_epoch = n_tr_epochs
		self.cliprange = cliprange
		self.model = None

		self.obs = None


		self.name = name

	def init_model(self, ob_space, ac_space, policy=policies.CnnPolicy, ent_coef=0.01):

		self.model = ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
								nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=self.ent, vf_coef=1,
								max_grad_norm=0.5, name=self.name)
		self.obs = np.zeros((1,) + ob_space.shape, dtype=self.model.train_model.X.dtype.name)

	def init_seed(self):
		# self.exp = 0
		# self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		pass

	def init_trajectory(self, start_cell=None, grid=None):
		# if start_cell.restore is not None:
		# 	if self.env.unwrapped.spec._env_name != "NChain":
		# 		(full_state, state, score, steps, pos, room_time, ram_death_state, _, _) = start_cell.restore
		# 		self.env.unwrapped.restore_full_state(full_state)
		# 		for i in range(3):  # TODO this puts the env out of sync
		# 			self.env.step(0)  # perform 3(4) nop to fill FrameStack
		# 	else:
		# 		state, _, _, _ = start_cell.restore
		# 		self.env.unwrapped.state = state
		# 	self.obs[:], _, self.done, _ = self.env.step(0)
		# else:
		# 	self.obs[:] = self.env.reset()
		self.done = True

	def seen_state(self, e):

		self.exp += 1

		# self.obs[:], reward, self.done, _ = self.env.step(self.mb_actions[self.actor][-1].squeeze())
		self.obs[:] = e['observation']
		self.mb_rewards[self.actor].append(e['reward'])
		self.done += e['done']

		if self.exp >= self.nsteps:
			self.actor += 1
			self.exp = 0

			if self.actor != self.nacts:
				self.mb_obs.append([])
				self.mb_actions.append([])
				self.mb_values.append([])
				self.mb_neglogpacs.append([])
				self.mb_dones.append([])
				self.mb_rewards.append([])
			else:
				self.actor = 0

				self.mb_obs = np.asarray(self.mb_obs, dtype=self.obs.dtype, ).squeeze()
				self.mb_rewards = np.asarray(self.mb_rewards, dtype=np.float32).squeeze()
				self.mb_actions = np.asarray(self.mb_actions).squeeze()
				self.mb_values = np.asarray(self.mb_values, dtype=np.float32).squeeze()
				self.mb_neglogpacs = np.asarray(self.mb_neglogpacs, dtype=np.float32).squeeze()
				self.mb_dones = np.asarray(self.mb_dones, dtype=np.bool).squeeze()

				if self.nacts > 1:
					self.mb_obs = self.mb_obs.swapaxes(0, 1)
					self.mb_rewards = self.mb_rewards.swapaxes(0, 1)
					self.mb_actions = self.mb_actions.swapaxes(0, 1)
					self.mb_values = self.mb_values.swapaxes(0, 1)
					self.mb_neglogpacs = self.mb_neglogpacs.swapaxes(0, 1)
					self.mb_dones = self.mb_dones.swapaxes(0, 1)

				# From baselines' ppo2 runner():
				# Calculate returns

				last_values = self.model.value(self.obs)

				mb_returns = np.zeros_like(self.mb_rewards)
				mb_advs = np.zeros_like(self.mb_rewards)
				lastgaelam = 0
				for t in reversed(range(self.nsteps)):
					if t == self.nsteps - 1:
						nextnonterminal = 1.0 - self.done
						nextvalues = last_values
					else:
						nextnonterminal = 1.0 - self.mb_dones[t + 1]
						nextvalues = self.mb_values[t + 1]
					delta = self.mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.mb_values[t]
					mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

				mb_returns[:] = mb_advs + self.mb_values
				# Swap and flatten axis 0 and 1
				if self.nacts > 1:
					self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs = \
						map(ppo2.sf01, (
						self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))

				# train model for multiple epoch in n minibacthes pr epoch
				# From baselines' ppo2 learn()
				inds = np.arange(self.batch)
				for _ in range(self.n_train_epoch):
					np.random.shuffle(inds)
					for start in range(0, self.batch, self.nbatch_train):
						end = start + self.nbatch_train
						mbinds = inds[start:end]
						slices = (arr[mbinds] for arr in (
						self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))
						self.model.train(self.lr, self.cliprange, *slices)

				self.lr *= self.lr_decay
				self.cliprange *= self.cl_decay

				self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [
																													   []], [
																													   []], [
																													   []], [
																													   []], [
																													   []], [
																													   []]

	def get_action(self, state, warmup_done):

		self.obs[:] = state

		actions, values, self.states, neglogpacs = self.model.step(self.obs)
		if warmup_done:
			self.mb_obs[self.actor].append(self.obs.copy())
			self.mb_actions[self.actor].append(actions)
			self.mb_values[self.actor].append(values)
			self.mb_neglogpacs[self.actor].append(neglogpacs)
			self.mb_dones[self.actor].append(self.done)

		return actions

	def clear(self):
		self.model.clear()

	def __repr__(self):
		return self.name



class DQNExplorer:
	def __init__(self, env):
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Conv2D(16, 8, input_shape=env.observation_space.shape, strides=4, activation=tf.nn.relu))
		self.model.add(tf.keras.layers.Conv2D(32, 4, strides=2, activation=tf.nn.relu))
		self.model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
		self.model.add(tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax))

		self.model.compile(optimizer=tf.keras.optimizers.Adam)


# class DQNet(tf.keras.Model):
#     def __init__(self, observation_sp, action_sp):
#         super(DQNet, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(16, 8, strides=4, activation=tf.nn.relu, input_shape=(observation_sp))
#         self.conv2 = tf.keras.layers.Conv2D(32, 4, strides=2, activation=tf.nn.relu)
#         self.dense = tf.keras.layers.Dense(256, activation=tf.nn.relu)
#         self.output = tf.keras.layers.Dense(action_sp, activation=tf.nn.softmax)

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.conv2(x)
#         x = self.dense(x)
#         return self.output(x)

class MlshExplorer_v2:
	def __init__(self, nsubs, timedialation, warmup_T, train_T,  actors, nexp, lr_mas, lr_sub, retrain_N = None,
				 lr_decay=1., cl_decay=1., lr_decay_sub=1., cl_decay_sub=1., nminibatches=4, n_tr_epochs=4,
				 cliprange_mas=0.1, cliprange_sub = 0.1, gamma=0.99, lam=0.95, ent_m=0.01, ent_s=0.01):





		# PPO related
		self.nacts = actors
		self.actor = 0
		self.exp = 0
		self.nsteps = nexp
		self.batch = self.nsteps * actors
		self.n_mb = nminibatches
		self.nbatch_train = self.batch // self.n_mb
		self.mb_obs_m, self.mb_rewards_m, self.mb_actions_m, self.mb_values_m, self.mb_dones_m, self.mb_neglogpacs_m, \
			self.mb_domains_m = [[]], [[]], [[]], [[]], [[]], [[]], [[]]
		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]], [[]], [[]], [[]], [[]], [[]]
		self.lr = lr_mas
		self.lr_decay = lr_decay
		self.cl_decay = cl_decay
		self.master_lr = lr_mas
		self.master_cl = cliprange_mas
		self.lr_sub = lr_sub
		self.cl_sub = cliprange_sub
		self.lr_dec_sub = lr_decay_sub
		self.cl_dec_sub = cl_decay_sub
		self.states = None
		self.done_m = [False for _ in range(1)]
		self.done = [False for _ in range(1)]
		self.gamma = gamma
		self.lam = lam
		self.ent = ent_m
		self.n_train_epoch = n_tr_epochs
		self.cliprange = cliprange_mas

		# MLSH
		self.master = None
		self.subs = None
		self.nsubs = nsubs
		self.time_dialation = timedialation
		self.warm_up_T = warmup_T
		self.warm_up_done = False
		self.train_t = train_T
		self.retrain_N = retrain_N
		self.cur_sub = None
		self.t = 0
		self.reward_m = 0
		self.reward = 0
		self.reset_count = 0


		self.obs = None
		self.domain = []
		self.env = None

	def init_model(self, env, domain_shape=None, masterPolicy=policies.CnnPolicy, subPolicies=policies.CnnPolicy):
		# self.env = gym.make(env)
		# if self.env.__repr__() != '<TimeLimit<NChainEnv<NChain-v0>>>':
		# 	self.env = ClipRewardEnv(FrameStack(WarpFrame(self.env), 4))
		# else:
		# 	self.env = self.env.unwrapped
		# 	self.env.unwrapped.n = 10000  #if nchain environment set N to 10 000
		# 	self.env = strechedObSpaceWrapper(self.env)
		# 	#TODO Should not be hardcoded
		# 	self.env.unwrapped.slip = 0

		ob_space = env.observation_space
		ac_space = gym.spaces.Discrete(self.nsubs)
		if  masterPolicy == policies.CnnPolicy_withDomain: #isinstance(masterPolicy., policies.CnnPolicy_withDomain):
			assert domain_shape is not None, Exception('domain policy but no domain shape suplied')
			self.master = ppo2.Model(policy=masterPolicy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
									 nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=self.ent, vf_coef=1,
									 max_grad_norm=0.5, name='Master', domain_shape=domain_shape)
			self.domain = np.zeros((1,) + domain_shape, dtype=self.master.train_model.G.dtype.name)
		else:
			self.master = ppo2.Model(policy=masterPolicy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
								nbatch_train=self.nbatch_train, nsteps=self.nsteps, ent_coef=self.ent, vf_coef=1,
								max_grad_norm=0.5, name='Master')

		self.subs = [ppo2.Model(policy=subPolicies, ob_space=ob_space, ac_space=env.action_space, nbatch_act=1,
								nbatch_train=self.time_dialation, nsteps=self.nsteps, ent_coef=0.01, vf_coef=1,
								max_grad_norm=0.5, name=f'Sub_{i}') for i in range(self.nsubs)]
		self.obs = np.zeros((1,) + ob_space.shape, dtype=self.master.train_model.X.dtype.name)


	def init_seed(self):
		# self.exp = 0
		# self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		pass

	def init_trajectory(self, obs, domain):

		# if start_cell.restore is not None:
		# 	if self.env.unwrapped.spec._env_name != "NChain":
		# 		(full_state, state, score, steps, pos, room_time, ram_death_state,_, _) = start_cell.restore
		# 		self.env.unwrapped.restore_full_state(full_state)
		# 		for i in range(3): #TODO this puts the env out of sync
		# 			self.env.step(0) #perform 3(4) nop to fill FrameStack
		# 	else:
		# 		state, _, _, _ = start_cell.restore
		# 		self.env.unwrapped.state = state
		# 	self.obs[:], _ , self.done, _ = self.env.step(0)
		# else:
		# 	self.obs[:] = self.env.reset()
		# 	self.done = [False]
		self.obs[:] = obs
		self.domain[:] = domain
		if len(self.mb_dones_m[self.actor]) > 0:
			self.mb_dones_m[self.actor][-1] = True # Delayed done flag to seperate from the previous episode which may not have ended in death

		if len(self.mb_dones[self.actor]) > 0:
			self.mb_dones[self.actor][-1] = True

		if self.t % self.time_dialation:
			self.t += (self.time_dialation - (self.t % self.time_dialation)) # fast-forward to next master time step
			#save partial exp
			self.mb_rewards_m[self.actor].append(self.reward_m)
			self.reward_m = 0
			self.exp += 1

			# if this complete a batch train before continuing
			if self.exp >= self.nsteps:
				self.actor += 1
				self.exp = 0

				if self.actor != self.nacts:
					self.mb_obs_m.append([])
					self.mb_actions_m.append([])
					self.mb_values_m.append([])
					self.mb_neglogpacs_m.append([])
					self.mb_dones_m.append([])
					self.mb_rewards_m.append([])
					self.mb_domains_m.append([])

					self.mb_obs.append([])
					self.mb_actions.append([])
					self.mb_values.append([])
					self.mb_neglogpacs.append([])
					self.mb_dones.append([])
					self.mb_rewards.append([])

				else:
					self.actor = 0
					if self.warm_up_done:
						self.train_subs()
					self.train()



	def seen_state(self, e):


		self.t += 1
		#self.obs[:], reward, self.done, _ = self.env.step(self.mb_actions[self.actor][-1].squeeze())

		self.obs[:] = e['observation']
		self.domain[:] = e['domain']
		self.done_m += e['done']
		self.done = e['done']
		self.reward_m += e['reward']
		self.reward = e['reward']

		if self.t % self.time_dialation == 0:

			self.mb_rewards_m[self.actor].append(self.reward_m)
			self.reward_m = 0
			self.exp += 1

		if self.warm_up_done:
				self.mb_rewards[self.actor].append(self.reward)



		if self.exp >= self.nsteps:
			self.actor += 1
			self.exp = 0



			if self.actor != self.nacts:
				self.mb_obs_m.append([])
				self.mb_actions_m.append([])
				self.mb_values_m.append([])
				self.mb_neglogpacs_m.append([])
				self.mb_dones_m.append([])
				self.mb_rewards_m.append([])
				self.mb_domains_m.append([])

				self.mb_obs.append([])
				self.mb_actions.append([])
				self.mb_values.append([])
				self.mb_neglogpacs.append([])
				self.mb_dones.append([])
				self.mb_rewards.append([])

			else:
				self.actor = 0
				if self.warm_up_done:
					self.train_subs()
				self.train()

	def train_subs(self):
		self.mb_obs = np.asarray(self.mb_obs, dtype=self.obs.dtype).squeeze()
		self.mb_rewards = np.asarray(self.mb_rewards, dtype=np.float32).squeeze()
		self.mb_actions = np.asarray(self.mb_actions).squeeze()
		self.mb_values = np.asarray(self.mb_values, dtype=np.float32).squeeze()
		self.mb_neglogpacs = np.asarray(self.mb_neglogpacs, dtype=np.float32).squeeze()
		self.mb_dones = np.asarray(self.mb_dones, dtype=np.bool).squeeze()

		if self.nacts > 1:
			self.mb_obs = self.mb_obs.swapaxes(0, 1)
			self.mb_rewards = self.mb_rewards.swapaxes(0, 1)
			self.mb_actions = self.mb_actions.swapaxes(0, 1)
			self.mb_values = self.mb_values.swapaxes(0, 1)
			self.mb_neglogpacs = self.mb_neglogpacs.swapaxes(0, 1)
			self.mb_dones = self.mb_dones.swapaxes(0, 1)

		last_values = self.subs[self.cur_sub].value(self.obs)

		mb_returns = np.zeros_like(self.mb_rewards)
		mb_advs = np.zeros_like(self.mb_rewards)
		lastgaelam = 0
		for t in reversed(range(len(self.mb_rewards))):
			if t == len(self.mb_rewards) - 1:
				nextnonterminal = 1.0 - self.done
				nextvalues = last_values
			else:
				nextnonterminal = 1.0 - self.mb_dones[t + 1]
				nextvalues = self.mb_values[t + 1]
			delta = self.mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.mb_values[t]
			mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

		mb_returns[:] = mb_advs + self.mb_values

		if self.nacts > 1:
			self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs= \
				map(ppo2.sf01, (self.mb_obs, mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_neglogpacs))
			master_actions = iter(ppo2.sf01(np.asarray(self.mb_actions_m).squeeze().swapaxes(0,1)))
		else:
			master_actions = iter(np.asarray(self.mb_actions_m).squeeze())

		subobs = [[] for _ in self.subs]
		subret = [[]for _ in self.subs]
		subdon = [[]for _ in self.subs]
		subact = [[] for _ in self.subs]
		subval = [[] for _ in self.subs]
		subneg = [[] for _ in self.subs]

		t = 0
		sub = int(next(master_actions))
		for i, _ in enumerate(mb_returns):

			subobs[sub].append(self.mb_obs[i])
			subret[sub].append(mb_returns[i])
			subdon[sub].append(self.mb_dones[i])
			subact[sub].append(self.mb_actions[i])
			subval[sub].append(self.mb_values[i])
			subneg[sub].append(self.mb_neglogpacs[i])

			t += 1
			if self.mb_dones[i] or t == self.time_dialation:
				t = 0
				try:
					sub = int(next(master_actions))
				except StopIteration:
					break

		subobs = [np.asarray(obs, dtype=self.obs.dtype) for obs in subobs]
		subret = [np.asarray(ret, dtype=np.float32).squeeze() for ret in subret]
		subact = [np.asarray(act).squeeze() for act in subact]
		subval = [np.asarray(val, dtype=np.float32).squeeze() for val in subval]
		subneg = [np.asarray(neg, dtype=np.float32).squeeze() for neg in subneg]
		subdon = [np.asarray(don, dtype=np.bool).squeeze() for don in subdon]

		for i in range(self.nsubs):
			inds = np.arange(subret[i].size)
			for _ in range(self.n_train_epoch):
				np.random.shuffle(inds)
				for start in range(0, subret[i].size, self.time_dialation):
					end = start + self.time_dialation
					if end > subret[i].size:
						continue
					mbinds = inds[start:end]
					slices = (arr[mbinds] for arr in (
						subobs[i], subret[i], subdon[i], subact[i], subval[i],
						subneg[i]))
					slices = dict(
						zip(['obs', 'returns', 'masks', 'actions', 'values', 'neglogpacs'], slices))
					self.subs[i].train(self.lr_sub, self.cl_sub, **slices)

		self.lr_sub *= self.lr_dec_sub
		self.cl_sub *= self.cl_dec_sub

		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]], [[]], [[]], [[]], [[]], [[]]
		pass



	def train(self):
		self.mb_obs_m = np.asarray(self.mb_obs_m, dtype=self.obs.dtype).squeeze()
		self.mb_rewards_m = np.asarray(self.mb_rewards_m, dtype=np.float32).squeeze()
		self.mb_actions_m = np.asarray(self.mb_actions_m).squeeze()
		self.mb_values_m = np.asarray(self.mb_values_m, dtype=np.float32).squeeze()
		self.mb_neglogpacs_m = np.asarray(self.mb_neglogpacs_m, dtype=np.float32).squeeze()
		self.mb_dones_m = np.asarray(self.mb_dones_m, dtype=np.bool).squeeze()
		if isinstance(self.master.train_model, policies.CnnPolicy_withDomain):
			self.mb_domains_m = np.asarray(self.mb_domains_m, dtype=self.domain.dtype).squeeze(axis=(0, 2))
		else:
			self.mb_domains_m = np.asarray(self.mb_domains_m).squeeze(axis=0)

		if self.nacts > 1:
			self.mb_obs_m = self.mb_obs_m.swapaxes(0, 1)
			self.mb_rewards_m = self.mb_rewards_m.swapaxes(0, 1)
			self.mb_actions_m = self.mb_actions_m.swapaxes(0, 1)
			self.mb_values_m = self.mb_values_m.swapaxes(0, 1)
			self.mb_neglogpacs_m = self.mb_neglogpacs_m.swapaxes(0, 1)
			self.mb_dones_m = self.mb_dones_m.swapaxes(0, 1)
			if isinstance(self.master.train_model, policies.CnnPolicy_withDomain):
				self.mb_domains_m = self.mb_domains_m.swapaxes(0, 1)

		# From baselines' ppo2 runner():
		# Calculate returns

		last_values = self.master.value(self.obs, self.domain)

		mb_returns = np.zeros_like(self.mb_rewards_m)
		mb_advs = np.zeros_like(self.mb_rewards_m)
		lastgaelam = 0
		for t in reversed(range(self.nsteps)):
			if t == self.nsteps - 1:
				nextnonterminal = 1.0 - self.done_m
				nextvalues = last_values
			else:
				nextnonterminal = 1.0 - self.mb_dones_m[t + 1]
				nextvalues = self.mb_values_m[t + 1]
			delta = self.mb_rewards_m[t] + self.gamma * nextvalues * nextnonterminal - self.mb_values_m[t]
			mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

		mb_returns[:] = mb_advs + self.mb_values_m
		# Swap and flatten axis 0 and 1
		if self.nacts > 1:
			self.mb_obs_m, mb_returns, self.mb_dones_m, self.mb_actions_m, self.mb_values_m, self.mb_neglogpacs_m, self.mb_domains_m = \
				map(ppo2.sf01, (self.mb_obs_m, mb_returns, self.mb_dones_m, self.mb_actions_m, self.mb_values_m, self.mb_neglogpacs_m, self.mb_domains_m))


		#train model for multiple epoch in n minibacthes pr epoch
		#From baselines' ppo2 learn()
		inds = np.arange(self.batch)
		for _ in range(self.n_train_epoch):
			np.random.shuffle(inds)
			for start in range(0, self.batch, self.nbatch_train):
				end = start + self.nbatch_train
				mbinds = inds[start:end]
				slices = (arr[mbinds] for arr in (self.mb_obs_m, mb_returns, self.mb_dones_m, self.mb_actions_m, self.mb_values_m, self.mb_neglogpacs_m, self.mb_domains_m))
				slices = dict(zip(['obs', 'returns', 'masks', 'actions', 'values', 'neglogpacs', 'domains'], slices))
				if not isinstance(self.master.train_model, policies.CnnPolicy_withDomain):
					slices['domains'] = None
				self.master.train(self.lr, self.cliprange, **slices)





		self.lr *= self.lr_decay
		self.cliprange *=  self.cl_decay

		self.mb_obs_m, self.mb_rewards_m, self.mb_actions_m, self.mb_values_m, self.mb_dones_m, self.mb_neglogpacs_m, \
		self.mb_domains_m = [[]], [[]], [[]], [[]], [[]], [[]], [[]]

	def get_action(self, state, env):

		#self.obs[:] = env.obs
		if (not self.warm_up_done) and self.t >= self.warm_up_T:
			self.warm_up_done = True
			self.t = 0

		if self.warm_up_done and self.t >= self.train_t:

			if self.retrain_N is not None and self.reset_count >= self.retrain_N:
				self.t = 0 # assuming subPolicies have convergede enough that no further resets are needed
			else:
				self.reset_master()
				self.t = 0
				self.warm_up_done = False # reset master policy and reenter warmup period

		if self.t % self.time_dialation == 0:
			self.cur_sub, master_values, _, master_neglogpacs = self.master.step(self.obs, self.domain)
			self.cur_sub = self.cur_sub.squeeze()


			self.mb_obs_m[self.actor].append(self.obs.copy())
			self.mb_domains_m[self.actor].append((self.domain.copy()))
			self.mb_actions_m[self.actor].append(self.cur_sub)
			self.mb_values_m[self.actor].append(master_values)
			self.mb_neglogpacs_m[self.actor].append(master_neglogpacs)
			self.mb_dones_m[self.actor].append(self.done_m)
			self.done_m = False

		actions, values, _, neglogpacs = self.subs[self.cur_sub].step(self.obs)

		if self.warm_up_done:
			self.mb_obs[self.actor].append(self.obs.copy())
			self.mb_actions[self.actor].append(actions)
			self.mb_values[self.actor].append(values)
			self.mb_neglogpacs[self.actor].append(neglogpacs)
			self.mb_dones[self.actor].append(self.done)

		return actions

	def reset_master(self):
		self.mb_obs_m, self.mb_rewards_m, self.mb_actions_m, self.mb_values_m, self.mb_dones_m, self.mb_neglogpacs_m = [[]], [[]], [[]], [[]], [[]], [[]]
		self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [[]], [[]], [[]], [[]], [[]], [[]]
		self.exp = 0

		self.actor = 0
		self.cliprange = self.master_cl
		self.lr = self.master_lr
		self.master.reset()



	def __repr__(self):
		return 'mlsh'


class GoalCondEnv:
	def __init__(self, MsPacmanEnv):
		self.env = MsPacmanEnv
		self.done = True
		self.timeLimit = 500

	def __getattr__(self, e):
		return getattr(self.env, e)

	def setGoal(self, startRestore, goalPos):
		self.startRestore = startRestore
		self.goalPos = goalPos

	def reset(self):
		self.done = False
		self.t = 0
		return self.env.restore(self.startRestore)

	def step(self, ac):
		self.t +=1
		if not self.done:
			obs, rew, done, lol = self.env.step(ac)

			if not done:
				if self.env.pos.level == self.goalPos.level:
					distance = np.sqrt((self.env.pos.x -self.goalPos.x)**2 + (self.env.pos.y - self.goalPos.y)**2)
					rew = -distance/100
					if distance < 4:

						done = True

				else:
					rew = -10
			else:
				rew = -10*self.timeLimit
			if self.t == self.timeLimit:
				done = True
			self.done = done
		else:
			obs = self.env._get_ob()
			rew = 0
			done = True
			lol = None


		return obs, rew, done, lol


def rand_selector(grid, ngoals):
	cell_keys = list( grid.keys() )
	return np.random.choice(cell_keys, ngoals)

class MlshExplorer_v3:
	def __init__(self, nsubs, timedialation, warmup_T, train_T,  actors, nexp=2000, lr=3e-5, retrain_N = None,
				 lr_decay=1., cl_decay=1., nminibatches=15, n_tr_epochs=10,
				 cliprange=0.2,  gamma=0.99, lam=0.98,
				 train_goals_pr_it=1, goal_selector=rand_selector):





		# PPO related
		self.nsteps = nexp
		self.batch = self.nsteps * actors
		self.lr = lr
		self.lr_decay = lr_decay
		self.cl_decay = cl_decay




		self.states = None
		self.gamma = gamma
		self.lam = lam
		self.n_train_epoch = n_tr_epochs
		self.cliprange = cliprange
		self.nBatches = nminibatches

		# MLSH
		self.master = None
		self.subs = None
		self.nsubs = nsubs
		self.time_dialation = timedialation
		self.warm_up_T = warmup_T
		self.warm_up_done = False
		self.train_t = train_T
		self.retrain_N = retrain_N
		self.cur_sub = None
		self.t = 0
		self.reward_m = 0
		self.reward = 0
		self.reset_count = 0


		self.obs = None
		self.domain = []
		self.env = None

		self.tr_pr_it = train_goals_pr_it
		self.goal_selector = goal_selector




	def init_model(self, env, masterPolicy=policies.MlshPolicy, subPolicies=policies.MlshPolicy):


		self.env = GoalCondEnv(env) #TODO should be made as a propper wrapper in future version but is not applicable
									# because the go-explore evironments arent made as wrapper

		ob_space = env.observation_space
		ac_space = gym.spaces.Discrete(self.nsubs)
		sess = tf.get_default_session()

		nh, nw, nc = ob_space.shape
		ob_shape = (None, nh, nw, nc)
		ob = U.get_placeholder(dtype=tf.uint8, shape=ob_shape, name='ob')  # obs

		self.master = masterPolicy( sess=sess, ob=ob, ac_space=ac_space,  reuse=False, name='Master')
		self.old_master = masterPolicy(sess=sess, ob=ob, ac_space=ac_space, reuse=False, name='old_Master')


		self.subs = [subPolicies(sess=sess, ob=ob, ac_space=env.action_space, name=f'Sub_{i}') for i in range(self.nsubs)]
		self.old_subs = [subPolicies(sess=sess, ob=ob, ac_space=env.action_space, name=f'old_Sub_{i}') for i in range(self.nsubs)]

		self.obs = np.zeros((1,) + ob_space.shape, dtype=self.master.X.dtype.name)

		# The comm object is needed for the mlsh learner, but should have no function when there is only one rank
		rank = MPI.COMM_WORLD.Get_rank()
		world_group = MPI.COMM_WORLD.Get_group()
		mygroup = rank % 10
		theta_group = world_group.Incl([x for x in range(MPI.COMM_WORLD.size) if (x % 10 == mygroup)])
		comm = MPI.COMM_WORLD.Create(theta_group)
		comm.Barrier()
		self.learner = Learner(self.env, self.master, self.old_master, self.subs, self.old_subs, comm, clip_param=self.cliprange, entcoeff=0,
						  optim_epochs=self.n_train_epoch, optim_stepsize=self.lr, optim_batchsize=64)
		args = lambda :None
		args.replay = False
		args.force_subpolicy = None
		self.rollout = rollouts.traj_segment_generator(self.master, self.subs, self.env, self.time_dialation, horizon=self.nsteps,
												  stochastic=True, args=args)

		self.learner.syncSubpolicies()


	def init_seed(self):
		# self.exp = 0
		# self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_neglogpacs = [],[],[],[],[],[]
		pass

	def init_trajectory(self, obs, domain):
		pass



	def seen_state(self, e):
		pass





	def train(self, startRestores, goalPoss):
		train_means = []
		for start, goal in zip(startRestores, goalPoss):
			self.env.setGoal(start,goal)
			self.master.reset()
			self.learner.syncMasterPolicies()

			mini_ep = 0
			totalmeans = []
			while mini_ep < self.warm_up_T + self.train_t:
				mini_ep += 1
				# rollout
				rolls = self.rollout.__next__()
				allrolls = []
				allrolls.append(rolls)
				# train theta
				rollouts.add_advantage_macro(rolls, self.time_dialation, self.gamma, self.lam)
				gmean, lmean = self.learner.updateMasterPolicy(rolls)
				# train phi
				test_seg = rollouts.prepare_allrolls(allrolls, self.time_dialation, self.gamma, self.lam, num_subpolicies=self.nsubs)
				self.learner.updateSubPolicies(test_seg, num_batches=self.nBatches, optimize=(mini_ep >= self.warm_up_T))
				# learner.updateSubPolicies(test_seg,
				# log
				totalmeans.append(gmean)
			train_means.append(totalmeans)
		return train_means



	def get_action(self, state, env):

		self.obs[:] = state
		if self.t  == 0:
			self.cur_sub = np.random.random_integers(0,self.nsubs-1)

		actions,_,_,_ = self.subs[self.cur_sub].step(self.obs)

		self.t += 1
		self.t %= self.time_dialation
		return actions



	def __repr__(self):
		return 'sample_mlsh'
