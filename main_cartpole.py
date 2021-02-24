from agents.PPO_cartpole import PPO
from runner.runner import Runner
import os
import time
import tensorflow as tf
from unity_env_wrapper import UnityEnvWrapper
import argparse
import numpy as np
import math
import gym

from reward_model.reward_model import RewardModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='grid')
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-sf', '--save-frequency', help="How many episodes after save the model", default=3000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=50)
parser.add_argument('-se', '--sampled-env', help="IRL", default=20)
parser.add_argument('-rc', '--recurrent', dest='recurrent', action='store_true')

# Parse arguments for Inverse Reinforcement Learning
parser.add_argument('-irl', '--inverse-reinforcement-learning', dest='use_reward_model', action='store_true')
parser.add_argument('-rf', '--reward-frequency', help="How many episode before update the reward model", default=15)
parser.add_argument('-rm', '--reward-model', help="The name of the reward model", default='warrior_10')
parser.add_argument('-dn', '--dems-name', help="The name of the demonstrations file", default='dems_archer.pkl')
parser.add_argument('-fr', '--fixed-reward-model', help="Whether to use a trained reward model",
                    dest='fixed_reward_model', action='store_true')

parser.set_defaults(use_reward_model=False)
parser.set_defaults(fixed_reward_model=False)
# TODO: use LSTM as default
parser.set_defaults(recurrent=False)

args = parser.parse_args()

eps = 1e-12

class GridWorld:

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self._max_episode_timesteps = 200
        self.graphics = False

    def reset(self):
        if self.graphics:
            self.env.render('human')
        state = self.env.reset()
        state = np.reshape(state, [4])
        #state *= [1.0, 0., 1., 0.]
        return dict(global_in=state)

    def execute(self, actions):
        if self.graphics:
            self.env.render('human')
        state, reward, done, _ = self.env.step(actions)
        state = np.reshape(state, [4])
        #state *= [1.0, 0., 1., 0.]
        state = dict(global_in=state)
        return state, done, reward

    def set_config(self, config):
        return

    def entropy(self, probs):
        entropy = 0
        for prob in probs:
            entropy += (prob + eps) * (math.log(prob + eps) + eps)

        return -entropy


if __name__ == "__main__":

    # DeepCrawl arguments
    model_name = args.model_name
    work_id = int(args.work_id)
    save_frequency = int(args.save_frequency)
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)
    sampled_env = int(args.sampled_env)
    # IRL
    use_reward_model = args.use_reward_model
    reward_model_name = args.reward_model
    fixed_reward_model = args.fixed_reward_model
    dems_name = args.dems_name
    reward_frequency = int(args.reward_frequency)

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = None

    # Total episode of training
    total_episode = 1e10
    # Units of training (episodes or timesteps)
    frequency_mode = 'episodes'
    # Frequency of training (in episode)
    frequency = 10
    # Memory of the agent (in episode)
    memory = 10

    # Create agent
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session(graph=graph)
        agent = PPO(sess=sess, memory=memory, model_name=model_name, recurrent=args.recurrent)
        # Initialize variables of models
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    # Open the environment with all the desired flags
    env = GridWorld()

    # If we want to use IRL, create a reward model
    reward_model = None
    if use_reward_model:
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            reward_sess = tf.compat.v1.Session(graph=graph)
            reward_model = RewardModel(actions_size=19, policy=agent, sess=reward_sess, name=model_name)
            # Initialize variables of models
            init = tf.compat.v1.global_variables_initializer()
            reward_sess.run(init)
            # If we want, we can use an already trained reward model
            if fixed_reward_model:
                reward_model.load_model(reward_model_name)
                print("Model loaded!")

    # Create runner
    runner = Runner(agent=agent, frequency=frequency, env=env, save_frequency=save_frequency,
                    logging=logging, total_episode=total_episode, curriculum=curriculum,
                    frequency_mode=frequency_mode,
                    reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                    fixed_reward_model=fixed_reward_model)
    try:
        runner.run()
    finally:
        #save_model(history, model_name, curriculum, agent)
        env.close()
