from agents.PPO_no_value_open_grid import PPO
from value_function.central_value_grid import CentralValue
import tensorflow as tf
import numpy as np

class MultiAgent:

    def __init__(self, num_agent, sess, recurrent=False, model_name='multi_agent', centralized_value_function=True,
                 memory = 10):
        self.num_agent = num_agent
        self.agents = []
        self.sess = sess
        self.model_name = model_name
        self.centralized_value_function = centralized_value_function
        self.memory = memory
        # Initialize all agents
        for i in range(self.num_agent):
            if self.centralized_value_function:
                self.agents.append(PPO(sess, action_type='discrete', action_size=7, model_name='ma_{}'.format(i),
                                   p_lr=3e-4, v_lr=3e-4, recurrent=recurrent, name='ma_{}'.format(i),
                                   memory=self.memory, p_num_itr=30, batch_fraction=0.1))
            else:
                print('OHOHOHOHO')
                self.agents.append(PPO_value(sess, action_type='continuous', action_size=2, model_name='ma_{}'.format(i),
                                       p_lr=5e-6, v_lr=5e-6, recurrent=recurrent, name='ma_{}'.format(i)))

        # Initialize value function
        if self.centralized_value_function:
            self.central_value = CentralValue(sess, v_lr=3e-4, memory=self.memory, v_num_itr=30, batch_fraction=0.1)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

    def train(self):
        if self.centralized_value_function:
            # First update the central value function
            self.central_value.train()

            # Then get the values from the central value function and pass them to the agents
            v_values = self.central_value.eval()

            # Train agents
            for ag in self.agents:
                ag.train(v_values)

            print("YAHAYAHA")
        else:
            for ag in self.agents:
                ag.train()

    # Eval each agents. Return an array of actions, logprobs and probs, one element for each agent
    def eval(self, state):
        a_actions = []
        a_logprobs = []
        a_probs = []

        for ag, s in zip(self.agents, state):
            act, logprobs, probs = ag.eval([s])
            a_actions.append(act[0])
            a_logprobs.append(logprobs)
            a_probs.append(probs)

        return a_actions, a_logprobs, a_probs

    # Add to buffer each experience of each agent AND a shared experience for the central value function
    # *REMEMBER that the reward function in this case is one for all the agents*
    # This method takes ARRAYS of experience
    def add_to_buffer(self, state, state_n, action, reward, old_prob, terminals,
                      internal_states_c=None, internal_states_h=None,
                      v_internal_states_c=None, v_internal_states_h=None):

        for (ag, s, s_n, a, o_p) in zip(self.agents, state, state_n, action, old_prob):

            ag.add_to_buffer(s, s_n, a, reward, o_p, terminals)

        if self.centralized_value_function:
            # Get the central value function state and pass it to the central value function buffer
            for i, s in enumerate(state):
                if i == 0:
                    central_value_state = [s['global_in']]
                else:
                    central_value_state = np.concatenate([central_value_state, [s['global_in']]])

            central_value_state = dict(global_in=central_value_state)
            # Add to buffer
            self.central_value.add_to_buffer(central_value_state, reward, terminals)


    # Save the entire model
    def save_model(self, name=None, folder='saved'):
        self.saver.save(self.sess, '{}/{}'.format(folder, name))
        return

    # Load entire model
    def load_model(self, name=None, folder='saved'):
        self.saver.restore(self.sess, '{}/{}'.format(folder, name))

        print('Model loaded correctly!')
        return
