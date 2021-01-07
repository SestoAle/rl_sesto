import os
import numpy as np
import json
from utils import NumpyEncoder
import math
import time

class Runner:
    def __init__(self, agent, frequency, save_frequency=3000, logging=100, total_episode=1e10, curriculum=None, **kwargs):

        # Runner objects and parameters
        self.agent = agent
        self.curriculum = curriculum
        self.total_episode = total_episode
        self.frequency = frequency
        self.logging = logging
        self.save_frequency = save_frequency

        # Global runner statistics
        # total episode
        self.ep = 0
        # total steps
        self.total_step = 0
        # Initialize history
        # History to save model statistics
        self.history = {
            "episode_rewards": [],
            "episode_timesteps": [],
            "mean_entropies": [],
            "std_entropies": [],
            "reward_model_loss": [],
            "env_rewards": []
        }

        # For curriculum training
        self.start_training = 0
        self.current_curriculum_step = 0

        # If a saved model with the model_name already exists, load it (and the history attached to it)
        if os.path.exists('{}/{}.meta'.format('saved', agent.model_name)):
            answer = None
            while answer != 'y' and answer != 'n':
                answer = input("There's already an agent saved with name {}, "
                               "do you want to continue training? [y/n] ".format(agent.model_name))

            if answer == 'y':
                history = self.load_model(agent.model_name, agent)
                self.ep = len(history['episode_timesteps'])
                self.total_step = np.sum(history['episode_timesteps'])


    def run(self, env):

        # Trainin loop
        # Start training
        start_time = time.time()
        while self.ep <= self.total_episode:
            # Reset the episode
            self.ep += 1
            step = 0

            # Set actual curriculum
            config = self.set_curriculum(self.curriculum, np.sum(self.history['episode_timesteps']))
            if self.start_training == 0:
                print(config)
            self.start_training = 1
            env.set_config(config)

            state = env.reset()
            done = False
            episode_reward = 0

            # Save local entropies
            local_entropies = []

            # Episode loop
            while True:

                # Evaluation - Execute step
                action, logprob, probs = self.agent.eval([state])

                action = action[0]
                # Save probabilities for entropy
                local_entropies.append(env.entropy(probs[0]))

                # Execute in the environment
                state_n, done, reward = env.execute(action)

                # If step is equal than max timesteps, terminate the episode
                if step >= env._max_episode_timesteps - 1:
                    done = True

                # Get the cumulative reward
                episode_reward += reward

                # Update PPO memory
                self.agent.add_to_buffer(state, state_n, action, reward, logprob, done)
                state = state_n

                step += 1
                self.total_step += 1

                # If done, end the episode and save statistics
                if done:
                    self.history['episode_rewards'].append(episode_reward)
                    self.history['episode_timesteps'].append(step)
                    self.history['mean_entropies'].append(np.mean(local_entropies))
                    self.history['std_entropies'].append(np.std(local_entropies))
                    break

            # Logging information
            if self.ep > 0 and self.ep % self.logging == 0:
                print('Mean of {} episode reward after {} episodes: {}'.
                      format(self.logging, self.ep, np.mean(self.history['episode_rewards'][-self.logging:])))

                print('The agent made a total of {} steps'.format(self.total_step))

                self.timer(start_time, time.time())

            # If frequency episodes are passed, update the policy
            if self.ep > 0 and self.ep % self.frequency == 0:
                total_loss = self.agent.train()

            # Save model and statistics
            if self.ep > 0 and self.ep % self.save_frequency == 0:
                self.save_model(self.history, self.agent.model_name, self.curriculum, self.agent)

    def save_model(self, history, model_name, curriculum, agent):

        # Save statistics as json
        json_str = json.dumps(history, cls=NumpyEncoder)
        f = open("arrays/{}.json".format(model_name), "w")
        f.write(json_str)
        f.close()

        # Save curriculum as json
        json_str = json.dumps(curriculum, cls=NumpyEncoder)
        f = open("arrays/{}_curriculum.json".format(model_name), "w")
        f.write(json_str)
        f.close()

        # Save the tf model
        agent.save_model(name=model_name, folder='saved')
        print('Model saved with name: {}'.format(model_name))

    def load_model(self, model_name, agent):
        agent.load_model(name=model_name, folder='saved')
        with open("arrays/{}.json".format(model_name)) as f:
            history = json.load(f)

        return history

    # Update curriculum for DeepCrawl
    def set_curriculum(self, curriculum, total_timesteps, mode='steps'):

        global current_curriculum_step

        if curriculum == None:
            return None

        if mode == 'steps':
            lessons = np.cumsum(curriculum['thresholds'])

            curriculum_step = 0

            for (index, l) in enumerate(lessons):

                if total_timesteps > l:
                    curriculum_step = index + 1

        parameters = curriculum['parameters']
        config = {}

        for (par, value) in parameters.items():
            config[par] = value[curriculum_step]

        current_curriculum_step = curriculum_step

        return config

    # Method for count time after each episode
    def timer(self, start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))