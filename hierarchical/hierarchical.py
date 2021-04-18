from agents.PPO import PPO
from agents.PPO_manager import PPO as PPOM
import tensorflow as tf
import numpy as np
import utils
import tensorflow_probability as tfp
eps = 1e-5

class HierarchicalAgent:

    def __init__(self, sess, manager_lr, workers_lr, num_workers, workers_name=None, model_name='hierarchical'):

        # Neural networks parameters
        self.manager_sess = sess
        self.manager_lr = manager_lr
        self.workers_lr = workers_lr
        self.model_name = model_name

        # Manager and Workers parameters
        self.manager_memory = 10
        self.manager_frequency = 5
        self.workers_memory = 5
        self.manager_timescale = 1
        self.workers_frequency = 5
        self.workers_action_size = 19
        # Whether workers are trainable or not
        self.trainable_workers = False
        self.continuous_manager = True

        self.manager = None
        self.workers = []

        # Instantiate the manager
        self.manager = PPO(
            self.manager_sess, memory=self.manager_memory, p_lr=self.manager_lr,  name='manager', action_size=num_workers,
            action_type='discrete', action_min_value=-1, action_max_value=1, num_workers=num_workers
        )

        # Instantiate the workers
        for i, w in enumerate(range(num_workers)):
            # Create agent
            graph = tf.compat.v1.Graph()
            with graph.as_default():
                tf.compat.v1.disable_eager_execution()
                sess = tf.compat.v1.Session(graph=graph)
                self.workers.append(
                    PPO(sess, memory=self.manager_memory, p_lr=self.manager_lr)#, name='worker_{}'.format(i))
                )

        if self.continuous_manager:
            # Load workers
            self.load_workers(workers_name)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

    def train(self, episode_count):

        if episode_count < 1:
            return

        # Train the manager with its memory after its frequency
        if episode_count % self.manager_frequency == 0:
            self.manager.train()

        if self.trainable_workers:
            # Train each of the workers with their memory after their frequency
            if episode_count % self.workers_frequency == 0:
                # If worker has enough experience
                for i, w in enumerate(self.workers):
                    if len(w.buffer['episode_lengths']) >= self.workers_frequency:
                        print('AHHAHHAHAHAHAHAH')
                        w.train()

    # Execute a manager action
    def eval_manager(self, state):
        man_action, man_logprob, man_probs = self.manager.eval(state)

        return man_action, man_logprob, man_probs

    # Execute a woker action specified by worker_id
    def eval_worker(self, state, man_actions):
        if not self.continuous_manager:
            work_action, work_logprob, work_probs = self.workers[man_actions].eval(state)
            return work_action, work_logprob, work_probs
        else:
            return self.eval_continuous_worker(state, man_actions)

    def add_terminations(self):
        # Add terminal to manager and update its episode_lengths
        self.manager.buffer['terminals'][-1] = True
        self.manager.buffer['episode_lengths'].append(
            int(len(self.manager.buffer['states']) - np.sum(self.manager.buffer['episode_lengths'])))
        # Add terminal to workers and update their episode_lengths
        for w in self.workers:
            if len(w.buffer['terminals']) > 0:
                w.buffer['terminals'][-1] = True
                w.buffer['episode_lengths'].append(
                    int(len(w.buffer['states']) - np.sum(w.buffer['episode_lengths'])))

    def eval_continuous_worker(self, state, man_actions):
        # Get the probability of each worker
        # man_actions += eps
        # # Softmax
        # man_actions = np.exp(man_actions) / np.sum(np.exp(man_actions), -1)
        # probs = np.zeros(self.workers_action_size)
        # for wg, wk in zip(man_actions, self.workers):
        #     work_action, work_logprob, work_probs = wk.eval(state)
        #     probs += (work_probs[0] * wg)
        #
        # #probs /= np.sum(man_actions)
        # probs = utils.boltzmann(probs, 1)
        # Sample action
        try:
            # action = np.argmax(np.random.multinomial(1, probs))
            # print(man_actions)
            action, log_probs, probs = self.workers[man_actions].eval(state)
            action = action[0]
            probs = probs[0]
        except Exception as e:
            #print(probs)
            print(e)
            print(man_actions)
            input('...')
        return [action], None, [probs]

    # Save the entire model
    def save_model(self, name=None, folder='saved'):
        self.saver.save(self.manager_sess, '{}/{}'.format(folder, name))
        return

    # Load entire model
    def load_model(self, name=None, folder='saved'):
        self.saver.restore(self.manager_sess, '{}/{}'.format(folder, name))

        print('Model loaded correctly!')
        return

    # Load each workers independently
    def load_workers(self, names=[], folder='saved'):
        if len(names) != len(self.workers):
            raise Exception('The length of the names must be equak to the number of workers')

        # Load each worker
        for i, n, w in zip(range(len(names)), names, self.workers):
            w.load_model(n, folder)
            print('Worker {} loaded correctly!'.format(i))
        return

    # Load only the manager
    def load_manager(self, name=None, folder='saved'):
        self.manager.load_model(name, folder)
        print('Manager loaded correctly!')
        return
