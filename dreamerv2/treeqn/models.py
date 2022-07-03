import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import keras.initializers as initializers

from dreamerv2.common import OneHotDist
from dreamerv2.treeqn.transitions import build_transition_fn, MLPRewardFn


class TreeQNPolicy(keras.Model):
    def __init__(self,
                 ob_space,
                 ac_space,
                 nenv,
                 nsteps,
                 nstack,
                 use_actor_critic=False,
                 transition_fun_name="matrix",
                 transition_nonlin="tanh",
                 normalise_state=True,
                 residual_transition=True,
                 tree_depth=2,
                 embedding_dim=512,
                 predict_rewards=True,
                 gamma=0.99,
                 td_lambda=0.8,
                 input_mode="atari",
                 value_aggregation="softmax",
                 output_tree=False):
        super(TreeQNPolicy, self).__init__()
        # nbatch = nenv * nsteps
        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nc * nstack, nh, nw)
        self.nenv = nenv
        self.num_actions = ac_space.n
        self.embedding_dim = embedding_dim
        self.use_actor_critic = use_actor_critic
        self.eps_threshold = 0
        self.predict_rewards = predict_rewards
        self.gamma = gamma
        self.output_tree = output_tree
        self.td_lambda = td_lambda
        self.normalise_state = normalise_state
        self.value_aggregation = value_aggregation

        self.embedding_dim = embedding_dim

        initializer_embed = initializers.Orthogonal(gain=2 ** 0.5)
        self.embed = layers.Dense(self.embedding_dim, activation='relu', kernel_initializer=initializer_embed)

        initializer_value = initializers.Orthogonal(gain=0.01)
        self.value_fn = layers.Dense(1, activation='linear', kernel_initializer=initializer_value)
        self.transition_nonlin = layers.Activation(keras.activations.tanh)
        self.transition_fun1, self.transition_fun2 = build_transition_fn(embedding_dim, num_actions=self.num_actions)

        if self.predict_rewards:
            self.tree_reward_fun = MLPRewardFn(embedding_dim, self.num_actions)

        self.tree_depth = tree_depth
        self.batch_size = 128

    def q(self, ob):
        shape = ob.shape
        return tf.reshape(self(tf.reshape(ob, shape=(-1, shape[-1])))[0], shape=(*shape[:2], -1))

    def actor(self, ob):
        return OneHotDist(logits=self(ob)[0], dtype=tf.dtypes.int32)

    def critic(self, ob):
        shape = ob.shape
        return tf.reshape(self(tf.reshape(ob, shape=(-1, shape[-1])))[1], shape=shape[:2])

    def call(self, observations):
        """
        :param ob: [batch_size x channels x height x width]
        :return: [batch_size x num_actions], -- Q-values
                 [batch_size x 1], -- V = max_a(Q)
                 [batch_size x num_actions x embedding_dim], -- embeddings after first transition
                 [batch_size x num_actions] -- rewards after first transition
        """

        Qs = []
        Vs = []
        for i in range(0, observations.shape[0], self.batch_size):
            ob = observations[i:i + self.batch_size]
            st = self.embed_obs(ob)

            if self.normalise_state:
                st = st / tf.sqrt(tf.reduce_sum(tf.pow(st, 2), axis=-1, keepdims=True))

            Q, tree_result = self.planning(st)
            V = tf.reduce_max(Q, axis=1)
            Qs.append(Q)
            Vs.append(V)

        return tf.concat(Qs, axis=0), tf.concat(Vs, axis=0), tree_result

    def embed_obs(self, ob):
        st = self.embed(ob)
        return st

    def step(self, ob):
        Q, V, _ = self(ob)
        a = self.sample(Q)
        return a, V

    def value(self, ob):
        _, V, _ = self(ob)
        return V

    def sample(self, Q):
        sample = random.random()
        if sample > self.eps_threshold:
            return tf.argmax(Q, axis=1).numpy()
        else:
            return np.random.randint(0, self.num_actions, self.nenv)

    def tree_planning(self, x, return_intermediate_values=True):
        """
        :param x: [batch_size x embedding_dim]
        :return:
            dict tree_result:
            - "embeddings":
                list of length tree_depth, [batch_size * num_actions^depth x embedding_dim] state
                representations after tree planning
            - "values":
                list of length tree_depth, [batch_size * num_actions^depth x 1] values predicted
                from each embedding
            - "rewards":
                list of length tree_depth, [batch_size * num_actions^depth x 1] rewards predicted
                from each transition
        """

        tree_result = {
            "embeddings": [x],
            "values": []
        }
        if self.predict_rewards:
            tree_result["rewards"] = []

        if return_intermediate_values:
            tree_result["values"].append(self.value_fn(x))

        for i in range(self.tree_depth):
            if self.predict_rewards:
                r = self.tree_reward_fun(x)
                tree_result["rewards"].append(tf.reshape(r, shape=(-1, 1)))

            x = self.tree_transitioning(x)

            x = tf.reshape(x, shape=(-1, self.embedding_dim))

            tree_result["embeddings"].append(x)

            if return_intermediate_values or i == self.tree_depth - 1:
                tree_result["values"].append(self.value_fn(x))

        return tree_result

    def tree_transitioning(self, x):
        """
        :param x: [? x embedding_dim]
        :return: [? x num_actions x embedding_dim]
        """
        x1 = self.transition_nonlin(self.transition_fun1(x))
        x2 = x + x1
        x2 = tf.expand_dims(x2, axis=1)
        x3 = self.transition_nonlin(tf.einsum("ij,jab->iba", x, self.transition_fun2))
        x2 = tf.repeat(x2, repeats=x3.shape[1], axis=1)
        next_state = x2 + x3

        if self.normalise_state:
            next_state = next_state / tf.sqrt(tf.reduce_sum(tf.pow(next_state, 2), axis=-1, keepdims=True))

        return next_state

    def planning(self, x):
        """
        :param x: [batch_size x embedding_dim] state representations
        :return:
            - [batch_size x embedding_dim x num_actions] state-action values
            - [batch_size x num_actions x embedding_dim] state representations after planning one step
              used for regularizing/grounding the transition model
        """
        batch_size = x.shape[0]
        if self.tree_depth > 0:
            tree_result = self.tree_planning(x)
        else:
            raise NotImplementedError

        q_values = self.tree_backup(tree_result, batch_size)

        return q_values, tree_result

    def tree_backup(self, tree_result, batch_size):
        backup_values = tree_result["values"][-1]
        for i in range(1, self.tree_depth + 1):
            one_step_backup = tree_result["rewards"][-i] + self.gamma * backup_values

            if i < self.tree_depth:
                one_step_backup = tf.reshape(one_step_backup, shape=(batch_size, -1, self.num_actions))

                if self.value_aggregation == "max":
                    max_backup = tf.reduce_max(one_step_backup, axis=2)
                elif self.value_aggregation == "softmax":
                    max_backup = tf.reduce_sum(one_step_backup * keras.activations.softmax(one_step_backup, axis=2),
                                               axis=2)
                else:
                    raise ValueError("Unknown value aggregation function %s" % self.value_aggregation)

                backup_values = ((1 - self.td_lambda) * tree_result["values"][-i - 1] +
                                 (self.td_lambda) * tf.reshape(max_backup, shape=(-1, 1)))
            else:
                backup_values = one_step_backup

        backup_values = tf.reshape(backup_values, shape=(batch_size, self.num_actions))

        return backup_values
