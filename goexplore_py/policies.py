#
# OBS
# This file is a deriviative of OpenAI's Baselines implementation
# Alteration made by:
# Jeppe Reinsborg, 3 June 2019
#
#
# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
import rl_algs.common.tf_util as U

def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """

    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

def domain_cnn(unscaled_images):


    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=4, stride=2, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    #h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h2)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, name='model'):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope(name, reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, name='model'):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope(name, reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)



        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, name='model'): #pylint: disable=W0613

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n

        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope(name, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        # run_metadata = tf.Run_Metadata()
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        def act(stoc, ob):
            a, v = sess.run([a0, vf], {X:ob})
            return a, v

        def get_variables(self):
            return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)

        def get_trainable_variables(self):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

        def reset(self):
            with tf.variable_scope(self.scope, reuse=True):
                varlist = self.get_trainable_variables()
                initializer = tf.variables_initializer(varlist)
                self.sess.run(initializer)


        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.sess = sess
        self.act = act
        self.get_variables = get_variables
        self.get_trainable_variables = get_trainable_variables
        self.reset = reset
        # self.metadata = run_metadata



class CnnPolicy_withDomain(object):

    def __init__(self, sess, ob_space, domain_shape , ac_space, nbatch, nsteps, reuse=False, name='model'): #pylint: disable=W0613

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        dh, dw, dc= domain_shape
        domain_shape = (nbatch, dh, dw, dc)
        nact = ac_space.n


        X = tf.placeholder(tf.uint8, ob_shape) #obs
        G = tf.placeholder(tf.uint8, domain_shape)
        with tf.variable_scope(name, reuse=reuse):
            with tf.variable_scope('obs', reuse=reuse):
                h = nature_cnn(X)
            with tf.variable_scope('domain', reuse=reuse):
                d = domain_cnn(G)
            c = tf.concat([h,d],1)
            pi = fc(c, 'pi', nact, init_scale=0.01)
            vf = fc(c, 'v', 1)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        # run_metadata = tf.Run_Metadata()
        # run_options =  tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        def step(ob, domain,  *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob, G:domain})
            return a, v, self.initial_state, neglogp

        def value(ob, domain, *_args, **_kwargs):
            return sess.run(vf, {X:ob, G:domain})

        self.X = X
        self.G = G
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        #self.metadata = run_metadata

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, name='model'): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.n #.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope(name, reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        #pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

# class MlshPolicy(object):
#     def __init__(self, sess, masterPolicy, subPolicy, nsub, ob_space, ac_space, nbatch, nsteps, reuse=False, name='MlshPolicy'):
#         self.master = masterPolicy(sess, ob_space, ac_space.__class__(nsub), nbatch, nsteps, reuse, name='Master')
#         self.sub = [subPolicy(sess, ob_space, ac_space, nbatch, nsteps, reuse, name=f'Sub{i}') for i in range(nsub)]
#
#         mA = self.master.pd.sample()
#
#         sA =
#
#     def step(self, ob):

class MlshPolicy(object):

    def __init__(self, sess, ob, ac_space, reuse=False, name='model'): #pylint: disable=W0613


        nact = ac_space.n



        with tf.variable_scope(name, reuse=reuse):
            self.scope = tf.get_variable_scope().name
            X = ob
            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:,0]

            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)




        # sample actions
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, vf])

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {ob:ob})

        self.X = X
        self.pi = pi
        self.selector = pi
        self.vf = vf
        self.vpred = vf
        self.step = step
        self.value = value
        self.sess = sess

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def reset(self):
        with tf.variable_scope(self.scope, reuse=True):
            varlist = self.get_trainable_variables()
            initializer = tf.variables_initializer(varlist)
            U.get_session().run(initializer)
