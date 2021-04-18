import os
import numpy as np
import json
from utils import NumpyEncoder
import time
from threading import Thread

# Act thread
class ActThreaded(Thread):
    def __init__(self, agent, env, parallel_buffer, index, config, num_steps, states):
        self.env = env
        self.parallel_buffer = parallel_buffer
        self.index = index
        self.env.set_config(config)
        self.num_steps = num_steps
        self.agent = agent
        self.states = states
        self.start()
        super().__init__()

    def run(self) -> None:
        state = self.states[self.index]
        for i in range(self.num_steps):
            # Execute the environment with the action
            actions, logprobs, probs = self.agent.eval([state])
            state_n, done, reward = self.env.execute(actions)

            if i == self.num_steps - 1:
                done = 2

            self.parallel_buffer[self.index].append(state)
            self.parallel_buffer['states'][self.index].append(state)
            self.parallel_buffer['states_n'][self.index].append(state_n)
            self.parallel_buffer['done'][self.index].append(done)
            self.parallel_buffer['reward'][self.index].append(reward)
            self.parallel_buffer['action'][self.index].append(actions)
            self.parallel_buffer['logprob'][self.index].append(logprobs)
            state = state_n
            if done:
                state = self.env.reset()

        if not done:
            self.parallel_buffer['done'][self.index].append(2)
        self.states[self.index] = state

# Epsiode thread
class EpisodeThreaded(Thread):
    def __init__(self, env, parallel_buffer, agent, index, config, num_episode=1, recurrent=False):
        self.env = env
        self.parallel_buffer = parallel_buffer
        self.agent = agent
        self.index = index
        self.num_episode = num_episode
        self.recurrent = recurrent
        self.env.set_config(config)
        super().__init__()

    def run(self) -> None:
        # Run each thread for num_episode episodes

        for i in range(self.num_episode):
            done = False
            step = 0
            # Reset the environment
            state = self.env.reset()

            # Total episode reward
            episode_reward = 0

            # Local entropies of the episode
            local_entropies = []

            # If recurrent, initialize hidden state
            if self.recurrent:
                internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))
                v_internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))

            while not done:
                # Evaluation - Execute step
                if not self.recurrent:
                    actions, logprobs, probs = self.agent.eval([state])
                else:
                    actions, logprobs, probs, internal_n, v_internal_n = self.agent.eval_recurrent([state], internal,
                                                                                                 v_internal)
                actions = actions[0]
                state_n, done, reward = self.env.execute(actions)

                #reward = reward[0]
                #done = done[0]

                episode_reward += reward
                local_entropies.append(self.env.entropy(probs[0]))
                # If step is equal than max timesteps, terminate the episode
                if step >= self.env._max_episode_timesteps - 1:
                    done = True
                self.parallel_buffer['states'][self.index].append(state)
                self.parallel_buffer['states_n'][self.index].append(state_n)
                self.parallel_buffer['done'][self.index].append(done)
                self.parallel_buffer['reward'][self.index].append(reward)
                self.parallel_buffer['action'][self.index].append(actions)
                self.parallel_buffer['logprob'][self.index].append(logprobs)

                if self.recurrent:
                    self.parallel_buffer['internal'][self.index].append(internal)
                    self.parallel_buffer['v_internal'][self.index].append(v_internal)
                    internal = internal_n
                    v_internal = v_internal_n

                state = state_n
                step += 1

            # History statistics
            self.parallel_buffer['episode_rewards'][self.index].append(episode_reward)
            self.parallel_buffer['episode_timesteps'][self.index].append(step)
            self.parallel_buffer['mean_entropies'][self.index].append(np.mean(local_entropies))
            self.parallel_buffer['std_entropies'][self.index].append(np.std(local_entropies))


class Runner:
    def __init__(self, agent, frequency, envs, save_frequency=3000, logging=100, total_episode=1e10, curriculum=None,
                 frequency_mode='episodes', random_actions=None, curriculum_mode='steps',
                 # IRL
                 reward_model=None, fixed_reward_model=False, dems_name='', reward_frequency=30,
                 # Adversarial Play
                 adversarial_play=False, double_agent=None,
                 **kwargs):

        # Runner objects and parameters
        self.agent = agent
        self.curriculum = curriculum
        self.total_episode = total_episode
        self.frequency = frequency
        self.frequency_mode = frequency_mode
        self.random_actions = random_actions
        self.logging = logging
        self.save_frequency = save_frequency
        self.envs = envs
        self.curriculum_mode = curriculum_mode

        # Recurrent
        self.recurrent = self.agent.recurrent

        # Objects and parameters for IRL
        self.reward_model = reward_model
        self.fixed_reward_model = fixed_reward_model
        self.dems_name = dems_name
        self.reward_frequency = reward_frequency

        # Adversarial play
        self.adversarial_play = adversarial_play
        self.double_agent = double_agent
        # If adversarial play, save the first version of the main agent and load it to the double agent
        if self.adversarial_play:
            self.agent.save_model(name=self.agent.model_name + '_0', folder='saved/adversarial')
            self.double_agent.load_model(name=self.agent.model_name + '_0', folder='saved/adversarial')

        # Initialize reward model
        if self.reward_model is not None:
            if not self.fixed_reward_model:
                # Ask for demonstrations
                answer = None
                while answer != 'y' and answer != 'n':
                    answer = input('Do you want to create new demonstrations? [y/n] ')
                if answer == 'y':
                    dems, vals = self.reward_model.create_demonstrations(env=self.envs[0])
                elif answer == 'p':
                    dems, vals = self.reward_model.create_demonstrations(env=self.envs[0], with_policy=True)
                else:
                    print('Loading demonstrations...')
                    dems, vals = self.reward_model.load_demonstrations(self.dems_name)

                print('Demonstrations loaded! We have ' + str(len(dems['obs'])) + " timesteps in these demonstrations")
                print('and ' + str(len(vals['obs'])) + " timesteps in these validations.")

                # Getting initial experience from the environment to do the first training epoch of the reward model
                self.get_experience(self.envs[0], self.reward_frequency, random=True)
                self.reward_model.train()

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

        # Initialize parallel buffer for savig experience of each thread without race conditions
        self.parallel_buffer = None
        self.parallel_buffer = self.clear_parallel_buffer()

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
                self.history = self.load_model(agent.model_name, agent)
                self.ep = len(self.history['episode_timesteps'])
                self.total_step = np.sum(self.history['episode_timesteps'])

    # Return a list of thread, that will save the experience in the shared buffer
    # The thread will run for 1 episode
    def create_episode_threads(self, parallel_buffer, agent, config):
        # The number of thread will be equal to the number of environments
        threads = []
        for i, e in enumerate(self.envs):
            # Create a thread
            threads.append(EpisodeThreaded(env=e, index=i, agent=agent, parallel_buffer=parallel_buffer, config=config,
                                           recurrent=self.recurrent))

        # Return threads
        return threads

    # Return a list of thread, that will save the experience in the shared buffer
    # The thread will run for 1 step of the environment
    def create_act_threds(self, agent, parallel_buffer, config, states, num_steps):
        # The number of thread will be equal to the number of environments
        threads = []
        for i, e in enumerate(self.envs):
            # Create a thread
            threads.append(ActThreaded(agent=agent, env=e, index=i, parallel_buffer=parallel_buffer, config=config,
                                       states=states, num_steps=num_steps))

        # Return threads
        return threads

    # Clear parallel buffer to avoid memory leak
    def clear_parallel_buffer(self):
        # Manually delete parallel buffer
        if self.parallel_buffer is not None:
            del self.parallel_buffer
        # Initialize parallel buffer for savig experience of each thread without race conditions
        parallel_buffer = {
            'states': [],
            'states_n': [],
            'done': [],
            'reward': [],
            'action': [],
            'logprob': [],
            'internal': [],
            'v_internal': [],
            # History
            'episode_rewards': [],
            'episode_timesteps': [],
            'mean_entropies': [],
            'std_entropies': [],

        }

        for i in range(len(self.envs)):
            parallel_buffer['states'].append([])
            parallel_buffer['states_n'].append([])
            parallel_buffer['done'].append([])
            parallel_buffer['reward'].append([])
            parallel_buffer['action'].append([])
            parallel_buffer['logprob'].append([])
            parallel_buffer['internal'].append([])
            parallel_buffer['v_internal'].append([])
            # History
            parallel_buffer['episode_rewards'].append([])
            parallel_buffer['episode_timesteps'].append([])
            parallel_buffer['mean_entropies'].append([])
            parallel_buffer['std_entropies'].append([])

        return parallel_buffer

    def run(self):

        # Trainin loop
        # Start training
        start_time = time.time()
        # If parallel act is in use, reset all environments at beginning of training

        if self.frequency_mode == 'timesteps':
            states = []
            for env in self.envs:
                states.append(env.reset())
        while self.ep <= self.total_episode:
            # Reset the episode
            # Set actual curriculum
            config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
            if self.start_training == 0:
                print(config)
            self.start_training = 1

            # Episode loop
            if self.frequency_mode=='episodes':
            # If frequency is episode, run the episodes in parallel
                # Create threads
                threads = self.create_episode_threads(agent=self.agent, parallel_buffer=self.parallel_buffer, config=config)

                # Run the threads
                for t in threads:
                    t.start()

                # Wait for the threads to finish
                for t in threads:
                    t.join()

                self.ep += len(threads)
            else:
            # If frequency is timesteps, run only the 'execute' in parallel for horizon steps

                # Create threads
                threads = self.create_act_threds(agent=self.agent, parallel_buffer=self.parallel_buffer, config=config,
                                                 states=states, num_steps=self.frequency)

                for t in threads:
                    t.start()

                for t in threads:
                    t.join()

                print(states[0])
                input('...')

                # Get how many episode are passed within threads
                print(self.parallel_buffer['dones'][:])
                input('...')



            # Add the overall experience to the buffer
            # Update the history
            for i in range(len(self.envs)):

                if not self.recurrent:
                    # Add to the agent experience in order of execution
                    for state, state_n, action, reward, logprob, done in zip(
                            self.parallel_buffer['states'][i],
                            self.parallel_buffer['states_n'][i],
                            self.parallel_buffer['action'][i],
                            self.parallel_buffer['reward'][i],
                            self.parallel_buffer['logprob'][i],
                            self.parallel_buffer['done'][i]
                    ):
                        self.agent.add_to_buffer(state, state_n, action, reward, logprob, done)
                else:
                    # Add to the agent experience in order of execution
                    for state, state_n, action, reward, logprob, done, internal, v_internal in zip(
                            self.parallel_buffer['states'][i],
                            self.parallel_buffer['states_n'][i],
                            self.parallel_buffer['action'][i],
                            self.parallel_buffer['reward'][i],
                            self.parallel_buffer['logprob'][i],
                            self.parallel_buffer['done'][i],
                            self.parallel_buffer['internal'][i],
                            self.parallel_buffer['v_internal'][i],
                    ):
                        try:
                            self.agent.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                     internal.c[0], internal.h[0], v_internal.c[0], v_internal.h[0])
                        except Exception as e:
                            zero_state = np.reshape(internal[0], [-1, ])
                            self.agent.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                     zero_state, zero_state, zero_state, zero_state)

                # Upadte the hisotry in order of execution
                for episode_reward, step, mean_entropies, std_entropies in zip(
                        self.parallel_buffer['episode_rewards'][i],
                        self.parallel_buffer['episode_timesteps'][i],
                        self.parallel_buffer['mean_entropies'][i],
                        self.parallel_buffer['std_entropies'][i],

                ):
                    self.history['episode_rewards'].append(episode_reward)
                    self.history['episode_timesteps'].append(step)
                    self.history['mean_entropies'].append(mean_entropies)
                    self.history['std_entropies'].append(std_entropies)

            # Clear parallel buffer
            self.parallel_buffer = self.clear_parallel_buffer()


            # Logging information
            if self.ep > 0 and self.ep % self.logging == 0:
                print('Mean of {} episode reward after {} episodes: {}'.
                      format(self.logging, self.ep, np.mean(self.history['episode_rewards'][-self.logging:])))

                if self.reward_model is not None:
                    print('Mean of {} environment episode reward after {} episodes: {}'.
                            format(self.logging, self.ep, np.mean(self.history['env_rewards'][-self.logging:])))

                print('The agent made a total of {} steps'.format(np.sum(self.history['episode_timesteps'])))

                self.timer(start_time, time.time())

            # If frequency episodes are passed, update the policy
            if self.frequency_mode == 'episodes' and self.ep > 0 and self.ep % self.frequency == 0:
                self.agent.train()

            # If IRL, update the reward model after reward_frequency episode
            if self.reward_model is not None:
                if not self.fixed_reward_model and self.ep > 0 and self.ep % self.reward_frequency == 0:
                    self.reward_model.train()

            # Save model and statistics
            if self.ep > 0 and self.ep % self.save_frequency == 0:
                self.save_model(self.history, self.agent.model_name, self.curriculum, self.agent)
                if self.reward_model is not None:
                    if not self.fixed_reward_model:
                        self.reward_model.save_model('{}_{}'.format(self.agent.model_name, self.ep))


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
    def set_curriculum(self, curriculum, history, mode='steps'):

        total_timesteps = np.sum(history['episode_timesteps'])
        total_episodes = len(history['episode_timesteps'])

        if curriculum == None:
            return None

        if mode == 'episodes':
            lessons = np.cumsum(curriculum['thresholds'])
            curriculum_step = 0

            for (index, l) in enumerate(lessons):

                if total_episodes > l:
                    curriculum_step = index + 1

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

        # If Adversarial play
        if self.adversarial_play:
            if curriculum_step > self.current_curriculum_step:
                # Save the current version of the main agent
                self.agent.save_model(name=self.agent.model_name + '_' + str(curriculum_step),
                                      folder='saved/adversarial')
                # Load the weights of the current version of the main agent to the double agent
                self.double_agent.load_model(name=self.agent.model_name + '_' + str(curriculum_step),
                                             folder='saved/adversarial')

        self.current_curriculum_step = curriculum_step

        return config

    # For IRL, get initial experience from environment, the agent act in the env without update itself
    def get_experience(self, env, num_discriminator_exp=None, verbose=False, random=False):

        if num_discriminator_exp == None:
            num_discriminator_exp = self.frequency

        # For policy update number
        for ep in range(num_discriminator_exp):
            states = []
            state = env.reset()
            step = 0
            # While the episode si not finished
            reward = 0
            while True:
                step += 1
                action, _, c_probs = self.agent.eval([state])
                if random:
                    num_actions = env.actions()['num_values']
                    action = np.random.randint(0, num_actions)
                state_n, terminal, step_reward = env.execute(actions=action)
                self.reward_model.add_to_buffer(state, state_n, action)

                state = state_n
                reward += step_reward
                if terminal or step >= env._max_episode_timesteps:
                    break

            if verbose:
                print("Reward at the end of episode " + str(ep + 1) + ": " + str(reward))

    # Method for count time after each episode
    def timer(self, start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
