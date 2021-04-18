import tensorflow as tf
import tensorflow_probability as tfp
import random
import numpy as np
from math import sqrt
import utils
from copy import deepcopy

import os

eps = 1e-5


# Actor-Critic PPO. The Actor is independent by the Critic.
class CentralValue:
    # PPO agent
    def __init__(self, sess, v_lr=5e-6, batch_fraction=0.33, v_num_itr=10,
                 discount=0.99, lmbda=1.0, name='central_value', memory=10, norm_reward=False,
                 model_name='central_value',

                 # LSTM
                 recurrent=False, recurrent_length=4, recurrent_baseline=False,

                 **kwargs):

        # Model parameters
        self.sess = sess
        self.v_lr = v_lr
        self.batch_fraction = batch_fraction
        self.v_num_itr = v_num_itr
        self.name = name
        self.norm_reward = norm_reward
        self.model_name = model_name

        # PPO hyper-parameters
        self.discount = discount
        self.lmbda = lmbda

        # Recurrent paramtere
        self.recurrent = recurrent
        self.recurrent_baseline = recurrent_baseline
        self.recurrent_length = recurrent_length
        self.recurrent_size = 256

        self.buffer = dict()
        self.clear_buffer()
        self.memory = memory
        # Create the network
        with tf.compat.v1.variable_scope(name) as vs:
            # Input spefication (for DeepCrawl)
            self.global_state = tf.compat.v1.placeholder(tf.float32, [None, 68], name='state')
            # Previous prob, for training
            self.old_logprob = tf.compat.v1.placeholder(tf.float32, [None, ], name='old_prob')
            self.baseline_values = tf.compat.v1.placeholder(tf.float32, [None, ], name='baseline_values')
            self.reward = tf.compat.v1.placeholder(tf.float32, [None, ], name='rewards')
            self.recurrent_train_length = tf.compat.v1.placeholder(tf.int32)
            self.sequence_lengths = tf.compat.v1.placeholder(tf.int32, [None, ])

            # Critic network
            with tf.compat.v1.variable_scope('critic'):

                # V Network specification
                self.v_network = self.conv_net(self.global_state, baseline=True)

                # Final p_layers
                if not self.recurrent_baseline:
                    self.v_network = self.linear(self.v_network, 256, name='v_fc1', activation=tf.nn.relu)
                else:
                    # The last FC layer will be replaced by an LSTM layer.
                    # Recurrent network needs more variables

                    # Get batch size and number of feature of the previous layer
                    bs, feature = utils.shape_list(self.v_network)
                    self.v_network = tf.reshape(self.v_network, [bs / self.recurrent_train_length,
                                                                 self.recurrent_train_length, feature])
                    # Define the RNN cell
                    self.v_rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=self.recurrent_size,
                                                                             state_is_tuple=True, activation=tf.nn.tanh)
                    # Define state_in for the cell
                    self.v_state_in = self.v_rnn_cell.zero_state(bs, tf.float32)

                    # Apply rnn
                    self.v_rnn, self.v_rnn_state = tf.compat.v1.nn.dynamic_rnn(
                        inputs=self.v_network, cell=self.v_rnn_cell, dtype=tf.float32, initial_state=self.v_state_in,
                        sequence_length=self.sequence_lengths
                    )

                    # Take only the last state of the sequence
                    self.v_network = self.v_rnn_state.h

                self.v_network = self.linear(self.v_network, 256, name='v_fc2', activation=tf.nn.relu)

                # Value function
                self.value = tf.squeeze(self.linear(self.v_network, 1))

            # Value function loss
            self.mse_loss = tf.compat.v1.losses.mean_squared_error(self.reward, self.value)

            # Value Optimizer
            self.v_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.v_lr).minimize(self.mse_loss)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

    ## Layers
    def linear(self, inp, inner_size, name='linear', bias=True, activation=None, init=None):
        with tf.compat.v1.variable_scope(name):
            lin = tf.compat.v1.layers.dense(inp, inner_size, name=name, activation=activation, use_bias=bias,
                                            kernel_initializer=init)
            return lin

    def conv_layer_2d(self, input, filters, kernel_size, strides=(2, 2), padding="SAME", name='conv',
                      activation=None, bias=True):

        with tf.compat.v1.variable_scope(name):
            conv = tf.compat.v1.layers.conv2d(input, filters, kernel_size, strides, padding=padding, name=name,
                                              activation=activation, use_bias=bias)
            return conv

    def embedding(self, input, indices, size, name='embs'):
        with tf.compat.v1.variable_scope(name):
            shape = (indices, size)
            stddev = min(0.1, sqrt(2.0 / (utils.product(xs=shape[:-1]) + shape[-1])))
            initializer = tf.random.normal(shape=shape, stddev=stddev, dtype=tf.float32)
            W = tf.Variable(
                initial_value=initializer, trainable=True, validate_shape=True, name='W',
                dtype=tf.float32, shape=shape
            )
            return tf.nn.tanh(tf.compat.v1.nn.embedding_lookup(params=W, ids=input, max_norm=None))

    # Convolutional network, the same for both policy and value networks
    def conv_net(self, global_state, baseline=False):
        global_state = self.linear(global_state, 1024, name='embs', activation=tf.nn.relu)
        return global_state

    def sample_batch_for_recurrent(self, length, batch_size):
        all_idxs = np.arange(len(self.buffer['states']))
        new_ep_lengths = deepcopy(self.buffer['episode_lengths'])

        for i, ep in enumerate(self.buffer['episode_lengths']):

            if ep < length:
                index = np.cumsum(new_ep_lengths)[i] - 1
                added_length = length - ep
                all_idxs = np.insert(all_idxs, index, np.ones(added_length) * all_idxs[index])
                new_ep_lengths[i] = length
            else:
                new_ep_lengths[i] = ep

        max_seq_steps = np.cumsum(new_ep_lengths) - length + 1
        min_seq_steps = np.concatenate([[0], np.cumsum(new_ep_lengths)])
        val_idxs = np.concatenate([np.arange(min, max) for min, max in zip(min_seq_steps, max_seq_steps)])

        batch_size = np.minimum(len(val_idxs), batch_size)

        val_idxs_first_step = np.random.choice(val_idxs, batch_size, replace=False)
        minibatch_idxs = []
        minibatch_idxs_last_step = []
        sequence_lengths = []
        minibatch_idxs_first_step = []
        for first in val_idxs_first_step:
            minibatch_idxs.extend(all_idxs[first:first + length])
            minibatch_idxs_last_step.append(all_idxs[first + length - 1])
            minibatch_idxs_first_step.append(all_idxs[first])
            parent_ep = np.sum(np.cumsum(self.buffer['episode_lengths']) <= all_idxs[first])

            sequence_lengths.append(np.minimum(length, self.buffer['episode_lengths'][parent_ep]))

        return minibatch_idxs, minibatch_idxs_last_step, minibatch_idxs_first_step, sequence_lengths

    # Train loop
    def train(self):
        v_losses = []

        # Get batch size based on batch_fraction
        batch_size = int(len(self.buffer['states']) * self.batch_fraction)
        if self.recurrent_baseline:
            batch_size = int(len(self.buffer['states']) * self.batch_fraction)

        # Before training, compute discounted reward
        discounted_rewards = self.compute_discounted_reward()

        # Train the value function
        for it in range(self.v_num_itr):
            if not self.recurrent_baseline:
                # Take a mini-batch of batch_size experience
                mini_batch_idxs = random.sample(range(len(self.buffer['states'])), batch_size)
                states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
            else:
                # Take the idxs of the sequences AND the idx of the last state of the sequence
                mini_batch_idxs, mini_batch_idxs_last_step, mini_batch_idxs_first_step, sequence_lengths = \
                    self.sample_batch_for_recurrent(self.recurrent_length, batch_size)
                states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
                v_internal_states_c = [self.buffer['v_internal_states_c'][id] for id in mini_batch_idxs_first_step]
                v_internal_states_h = [self.buffer['v_internal_states_h'][id] for id in mini_batch_idxs_first_step]
                tmp_batch_size = len(states_mini_batch) // self.recurrent_length
                v_internal_states_c = np.reshape(v_internal_states_c, [tmp_batch_size, -1])
                v_internal_states_h = np.reshape(v_internal_states_h, [tmp_batch_size, -1])
                v_internal_states = (v_internal_states_c, v_internal_states_h)
                mini_batch_idxs = mini_batch_idxs_last_step

            rewards_mini_batch = [discounted_rewards[id] for id in mini_batch_idxs]
            # Reshape problem, why?
            rewards_mini_batch = np.reshape(rewards_mini_batch, [-1, ])

            # Get DeepCrawl state
            # Convert the observation to states
            states = self.obs_to_state(states_mini_batch)
            feed_dict = self.create_state_feed_dict(states)

            # Update feed dict for training
            feed_dict[self.reward] = rewards_mini_batch

            if not self.recurrent_baseline:
                v_loss, step = self.sess.run([self.mse_loss, self.v_step], feed_dict=feed_dict)
            else:
                # If recurrent, we need to pass the internal state and the recurrent_length
                feed_dict[self.v_state_in] = v_internal_states
                feed_dict[self.sequence_lengths] = sequence_lengths
                feed_dict[self.recurrent_train_length] = self.recurrent_length
                v_loss, step = self.sess.run([self.mse_loss, self.v_step], feed_dict=feed_dict)

            v_losses.append(v_loss)

        return np.mean(v_losses)

    # Eval sampling the action (done by the net)
    def eval(self):
        states = self.obs_to_state(self.buffer['states'])
        feed_dict = self.create_state_feed_dict(states)
        if self.recurrent_baseline:
            v_internal_states_c = self.buffer['v_internal_states_c']
            v_internal_states_h = self.buffer['v_internal_states_h']
            v_internal_states_c = np.reshape(v_internal_states_c, [len(self.buffer['states']), -1])
            v_internal_states_h = np.reshape(v_internal_states_h, [len(self.buffer['states']), -1])
            v_internal_states = (v_internal_states_c, v_internal_states_h)
            feed_dict[self.v_state_in] = v_internal_states
            feed_dict[self.sequence_lengths] = np.ones(len(self.buffer['states']))
            feed_dict[self.recurrent_train_length] = 1

        v_values = self.sess.run(self.value, feed_dict=feed_dict)
        v_values = np.append(v_values, 0)

        return v_values

    # Eval sampling the action, but with recurrent: it needs the internal hidden state
    def eval_recurrent(self, state, internal, v_internal=None):
        state = self.obs_to_state(state)
        feed_dict = self.create_state_feed_dict(state)

        # Pass the internal state
        feed_dict[self.state_in] = internal
        feed_dict[self.recurrent_train_length] = 1
        feed_dict[self.sequence_lengths] = [1]
        action, logprob, probs, internal = self.sess.run([self.action, self.log_prob, self.probs, self.rnn_state],
                                                         feed_dict=feed_dict)
        if self.recurrent_baseline:
            feed_dict[self.v_state_in] = v_internal
            v_internal = self.sess.run([self.v_state_in], feed_dict=feed_dict)

        # Return is equal to eval(), but with the new internal state
        return action, logprob, probs, internal, v_internal

    # Transform an observation to a DeepCrawl state
    def obs_to_state(self, obs):
        global_batch = np.stack([np.asarray(state['global_in']) for state in obs])

        return global_batch

    # Create a state feed_dict from states
    def create_state_feed_dict(self, states):
        all_global = states

        feed_dict = {
            self.global_state: all_global
        }

        return feed_dict

    # Clear the memory buffer
    def clear_buffer(self):

        self.buffer['episode_lengths'] = []
        self.buffer['states'] = []
        self.buffer['rewards'] = []
        self.buffer['terminals'] = []
        if self.recurrent:
            self.buffer['v_internal_states_c'] = []
            self.buffer['v_internal_states_h'] = []

    # Add a transition to the buffer
    def add_to_buffer(self, state, reward, terminals,
                      v_internal_states_c=None, v_internal_states_h=None):

        # If we store more than memory episodes, remove the last episode
        if len(self.buffer['episode_lengths']) + 1 >= self.memory + 1:
            idxs_to_remove = self.buffer['episode_lengths'][0]
            del self.buffer['states'][:idxs_to_remove]
            del self.buffer['rewards'][:idxs_to_remove]
            del self.buffer['terminals'][:idxs_to_remove]
            del self.buffer['episode_lengths'][0]
            if self.recurrent:
                del self.buffer['internal_states_c'][:idxs_to_remove]
                del self.buffer['internal_states_h'][:idxs_to_remove]
            if self.recurrent_baseline:
                del self.buffer['v_internal_states_c'][:idxs_to_remove]
                del self.buffer['v_internal_states_h'][:idxs_to_remove]

        self.buffer['states'].append(state)
        self.buffer['rewards'].append(reward)
        self.buffer['terminals'].append(terminals)
        if self.recurrent_baseline:
            self.buffer['v_internal_states_c'].append(v_internal_states_c)
            self.buffer['v_internal_states_h'].append(v_internal_states_h)
        # If its terminal, update the episode length count (all states - sum(previous episode lengths)
        if terminals:
            self.buffer['episode_lengths'].append(
                int(len(self.buffer['states']) - np.sum(self.buffer['episode_lengths'])))

    # Change rewards in buffer to discounted rewards
    def compute_discounted_reward(self):

        discounted_rewards = []
        discounted_reward = 0
        # The discounted reward can be computed in reverse
        for (terminal, reward) in zip(reversed(self.buffer['terminals']), reversed(self.buffer['rewards'])):
            if terminal:
                discounted_reward = 0

            discounted_reward = reward + (self.discount * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalizing reward
        if self.norm_reward:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)

        return discounted_rewards

    # Change rewards in buffer to discounted rewards or GAE rewards (if lambda == 1, gae == discounted)
    def compute_gae(self, v_values):

        rewards = []
        gae = 0

        # The gae rewards can be computed in reverse
        for i in reversed(range(len(self.buffer['rewards']))):
            terminal = self.buffer['terminals'][i]
            m = 1
            if terminal:
                m = 0

            delta = self.buffer['rewards'][i] + self.discount * v_values[i + 1] * m - v_values[i]
            gae = delta + self.discount * self.lmbda * m * gae
            reward = gae + v_values[i]

            rewards.insert(0, reward)

        # Normalizing
        if self.norm_reward:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + eps)

        return rewards

    # Save the entire model
    def save_model(self, name=None, folder='saved'):
        self.saver.save(self.sess, '{}/{}'.format(folder, name))

        if False:
            graph_def = self.sess.graph.as_graph_def()

            # freeze_graph clear_devices option
            for node in graph_def.node:
                node.device = ''

            graph_def = tf.compat.v1.graph_util.remove_training_nodes(input_graph=graph_def)
            output_node_names = [
                'ppo/actor/add',
                'ppo/actor/ppo_actor_Categorical/action/Reshape_2',
                'ppo/critic/Squeeze'
            ]

            # implies tf.compat.v1.graph_util.extract_sub_graph
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess=self.sess, input_graph_def=graph_def,
                output_node_names=output_node_names
            )
            graph_path = tf.io.write_graph(
                graph_or_graph_def=graph_def, logdir=folder,
                name=(name + '.pb'), as_text=False
            )

        return

    # Load entire model
    def load_model(self, name=None, folder='saved'):
        # self.saver = tf.compat.v1.train.import_meta_graph('{}/{}.meta'.format(folder, name))
        self.saver.restore(self.sess, '{}/{}'.format(folder, name))

        print('Model loaded correctly!')
        return
