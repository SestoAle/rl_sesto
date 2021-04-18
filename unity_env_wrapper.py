from tensorforce.environments import Environment
import numpy as np
import math
import signal
import time
import logging

eps = 1e-12

# Load UnityEnvironment and my wrapper
from mlagents.envs import UnityEnvironment

class UnityEnvWrapper(Environment):
    def __init__(self, game_name = None, no_graphics = True, seed = None, worker_id=0, size_global = 8, size_two = 5,
                 with_local = True, size_three = 3, with_stats=True, size_stats = 1,
                 with_previous=True, manual_input = False, config = None, curriculum = None, verbose = False,
                 agent_separate = False, agent_stats = 6, action_size = 19,
                 _max_episode_timesteps = 100,
                 with_class=False, with_hp = False, size_class = 3,
                 # Double agent
                 use_double_agent=False, double_agent = None, play_with_random=0.33,
                 # Reward model
                 reward_model = None):

        self.probabilities = []

        self.size_global = size_global
        self.size_two = size_two
        self.with_local = with_local
        self.size_three = size_three
        self.with_stats = with_stats
        self.size_stats = size_stats
        self.action_size = action_size

        self.manual_input = manual_input
        self.with_previous = with_previous
        self.config = config
        self.curriculum = curriculum
        self.verbose = verbose

        self.agent_separate = agent_separate
        self.agent_stats = agent_stats
        self.with_class = with_class
        self.with_hp = with_hp
        self.size_class = size_class
        self.use_double_agent = use_double_agent

        self.game_name = game_name

        self.no_graphics = no_graphics
        self.seed = seed

        self.worker_id = worker_id
        self.unity_env = self.open_unity_environment(game_name, no_graphics, seed, worker_id)
        self.default_brain = self.unity_env.brain_names[0]

        self.input_channels = 6
        self.one_hot = True
        self.reward_model = reward_model
        self._max_episode_timesteps = _max_episode_timesteps

        # Adversarial play
        if(self.use_double_agent):
            self.double_brain = self.unity_env.brain_names[1]
            self.double_agent = double_agent
            self.play_with_random = play_with_random

        self.global_timesteps = 0


        self.with_transformer = False

    count = 0

    def to_one_hot(self, a, channels):
        return (np.arange(channels) == a[..., None]).astype(float)

    def get_input_observation(self, env_info, action=None, transformer=False):
        size = self.size_global * self.size_global * self.input_channels

        global_in = env_info.vector_observations[0][:size]
        global_in = np.reshape(global_in, (self.size_global, self.size_global, self.input_channels))
        if self.one_hot:
            global_in_one_hot = self.to_one_hot(global_in[:,:,0], 7)
            for i in range(1, self.input_channels):
                global_in_one_hot = np.append(global_in_one_hot, self.to_one_hot(global_in[:,:,i], 9), axis = 2)

        if self.with_local:
            size_local = self.size_two * self.size_two * self.input_channels
            local_in = env_info.vector_observations[0][size:(size + size_local)]
            local_in = np.reshape(local_in, (self.size_two, self.size_two, self.input_channels))
            if self.one_hot:
                local_in_one_hot = self.to_one_hot(local_in[:, :, 0], 7)
                for i in range(1, self.input_channels):
                    local_in_one_hot = np.append(local_in_one_hot, self.to_one_hot(local_in[:, :, i], 9), axis=2)

            size_local_two = self.size_three * self.size_three * self.input_channels
            local_in_two = env_info.vector_observations[0][(size + size_local):(
                    size + size_local + size_local_two)]
            local_in_two = np.reshape(local_in_two, (self.size_three, self.size_three, self.input_channels))
            if self.one_hot:
                local_in_two_one_hot = self.to_one_hot(local_in_two[:, :, 0], 7)
                for i in range(1, self.input_channels):
                    local_in_two_one_hot = np.append(local_in_two_one_hot, self.to_one_hot(local_in_two[:, :, i], 9), axis=2)

        if self.with_local and self.with_stats:
            stats = env_info.vector_observations[0][
                    (size + size_local + size_local_two):
                    (size + size_local + size_local_two + self.size_stats)]

        if self.with_local and self.with_stats and self.with_hp:
            hp = env_info.vector_observations[0][
                 (size + (self.size_two * self.size_two) + (self.size_three * self.size_three)):
                 (size + (self.size_two * self.size_two) + (self.size_three * self.size_three) + 4)]
            stats = env_info.vector_observations[0][
                    (size + (self.size_two * self.size_two) + (self.size_three * self.size_three) + 4):
                    (size + (self.size_two * self.size_two) + (
                                self.size_three * self.size_three) + self.size_stats + 4)]
            self.size_stats += 4

        if self.with_local and self.with_stats and self.with_class:
            agent_class = env_info.vector_observations[0][
                          (size + (self.size_two * self.size_two) + (
                                      self.size_three * self.size_three) + self.size_stats):
                          (size + (self.size_two * self.size_two) + (
                                      self.size_three * self.size_three) + self.size_stats + self.size_class)]

            enemy_class = env_info.vector_observations[0][
                          (size + (self.size_two * self.size_two) + (
                                      self.size_three * self.size_three) + self.size_stats + self.size_class):
                          (size + (self.size_two * self.size_two) + (
                                      self.size_three * self.size_three) + self.size_stats + self.size_class * 2)]

        observation = {
            'global_in': global_in,
            'local_in': local_in
        }

        if self.with_local:
            observation = {
                'global_in': global_in,
                'local_in': local_in,
                'local_in_two': local_in_two
            }
        if self.with_local and self.with_stats:
            observation = {
                'global_in': global_in,
                'local_in': local_in,
                'local_in_two': local_in_two,
                'stats': stats
            }

        if self.with_local and self.with_stats and self.with_previous:
            action_vector = np.zeros(19)
            if action != None:
                action_vector[action] = 1

            observation = {
                # Global View
                'global_in': global_in_one_hot,
                # 'attr_global_1': global_in[:, :, 1],
                # 'attr_global_2': global_in[:, :, 2],
                # 'attr_global_3': global_in[:, :, 3],
                # 'attr_global_4': global_in[:, :, 4],
                # 'attr_global_5': global_in[:, :, 5],
                #'global_in_attributes': global_in[:, :, 1:]/5.,

                # Local View
                'local_in': local_in_one_hot,
                # 'attr_local_1': local_in[:, :, 1],
                # 'attr_local_2': local_in[:, :, 2],
                # 'attr_local_3': local_in[:, :, 3],
                # 'attr_local_4': local_in[:, :, 4],
                # 'attr_local_5': local_in[:, :, 5],
                #'local_in_attributes': local_in[:, :, 1:]/5.,

                # Local Two View
                'local_in_two': local_in_two_one_hot,
                # 'attr_local_two_1': local_in_two[:, :, 1],
                # 'attr_local_two_2': local_in_two[:, :, 2],
                # 'attr_local_two_3': local_in_two[:, :, 3],
                # 'attr_local_two_4': local_in_two[:, :, 4],
                # 'attr_local_two_5': local_in_two[:, :, 5],
                #'local_in_two_attributes': local_in_two[:, :, 1:],

                # Stats
                'agent_stats': stats[:16],
                'target_stats': stats[16:],
                'prev_action': action_vector
            }

            if transformer:
                x_map = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

                y_map = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                     [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                                     [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                                     [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                                     [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                                     [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                                     [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                                     [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]])


                x_map = np.reshape(x_map, [10, 10, 1])
                y_map = np.reshape(y_map, [10, 10, 1])

                x_map = 2 * (x_map - 0) / (9 - 0) - 1
                y_map = 2 * (y_map - 0) / (9 - 0) - 1

                x_local_two_map = np.asarray([[0, 1, 2],
                                              [0, 1, 2],
                                              [0, 1, 2]])

                y_local_two_map = np.asarray([[0, 0, 0],
                                              [1, 1, 1],
                                              [2, 2, 2]])


                x_local_two_map = np.reshape(x_local_two_map, [3, 3, 1])
                y_local_two_map = np.reshape(y_local_two_map, [3, 3, 1])

                x_local_two_map = 2 * (x_local_two_map - 0) / (3 - 0) - 1
                y_local_two_map = 2 * (y_local_two_map - 0) / (3 - 0) - 1

                observation['global_positions'] = np.concatenate([x_map, y_map], axis=2)
                observation['local_two_positions'] = np.concatenate([x_local_two_map, y_local_two_map], axis=2)

        if self.with_local and self.with_stats and self.with_previous and self.with_class:
            action_vector = np.zeros(17)
            if action != None:
                action_vector[action] = 1

            observation = {
                'global_in': global_in,
                'local_in': local_in,
                'local_in_two': local_in_two,
                'stats': stats,
                'agent_class': agent_class,
                'enemy_class': enemy_class,
                'action': action_vector
            }

        if self.with_local and self.with_stats and self.with_previous and self.with_hp:
            action_vector = np.zeros(17)
            if action != None:
                action_vector[action] = 1

            observation = {
                'global_in': global_in,
                'local_in': local_in,
                'local_in_two': local_in_two,
                'hp': hp,
                'stats': stats,
                'prev_action': action_vector
            }

        return observation

    def get_input_observation_adapter(self, env_info, action = None):
        size = self.size_global * self.size_global * self.input_channels

        global_in = env_info.vector_observations[0][:size]
        global_in = np.reshape(global_in, (self.size_global, self.size_global, self.input_channels))
        if self.one_hot:
            global_in_one_hot = self.to_one_hot(global_in[:,:,0], 7)
            for i in range(1, self.input_channels):
                global_in_one_hot = np.append(global_in_one_hot, self.to_one_hot(global_in[:,:,i], 9), axis = 2)

        if self.with_local:
            size_local = self.size_two * self.size_two * self.input_channels
            local_in = env_info.vector_observations[0][size:(size + size_local)]
            local_in = np.reshape(local_in, (self.size_two, self.size_two, self.input_channels))
            if self.one_hot:
                local_in_one_hot = self.to_one_hot(local_in[:, :, 0], 7)
                for i in range(1, self.input_channels):
                    local_in_one_hot = np.append(local_in_one_hot, self.to_one_hot(local_in[:, :, i], 9), axis=2)

            size_local_two = self.size_three * self.size_three * self.input_channels
            local_in_two = env_info.vector_observations[0][(size + size_local):(
                    size + size_local + size_local_two)]
            local_in_two = np.reshape(local_in_two, (self.size_three, self.size_three, self.input_channels))
            if self.one_hot:
                local_in_two_one_hot = self.to_one_hot(local_in_two[:, :, 0], 7)
                for i in range(1, self.input_channels):
                    local_in_two_one_hot = np.append(local_in_two_one_hot, self.to_one_hot(local_in_two[:, :, i], 9), axis=2)

        if self.with_local and self.with_stats:
            stats = env_info.vector_observations[0][
                    (size + size_local + size_local_two):
                    (size + size_local + size_local_two + self.size_stats)]

        if self.with_local and self.with_stats and self.with_hp:
            hp = env_info.vector_observations[0][
                 (size + (self.size_two * self.size_two) + (self.size_three * self.size_three)):
                 (size + (self.size_two * self.size_two) + (self.size_three * self.size_three) + 4)]
            stats = env_info.vector_observations[0][
                    (size + (self.size_two * self.size_two) + (self.size_three * self.size_three) + 4):
                    (size + (self.size_two * self.size_two) + (
                                self.size_three * self.size_three) + self.size_stats + 4)]
            self.size_stats += 4

        if self.with_local and self.with_stats and self.with_class:
            agent_class = env_info.vector_observations[0][
                          (size + (self.size_two * self.size_two) + (
                                      self.size_three * self.size_three) + self.size_stats):
                          (size + (self.size_two * self.size_two) + (
                                      self.size_three * self.size_three) + self.size_stats + self.size_class)]

            enemy_class = env_info.vector_observations[0][
                          (size + (self.size_two * self.size_two) + (
                                      self.size_three * self.size_three) + self.size_stats + self.size_class):
                          (size + (self.size_two * self.size_two) + (
                                      self.size_three * self.size_three) + self.size_stats + self.size_class * 2)]

        observation = {
            'global_in': global_in,
            'local_in': local_in
        }

        if self.with_local:
            observation = {
                'global_in': global_in,
                'local_in': local_in,
                'local_in_two': local_in_two
            }
        if self.with_local and self.with_stats:
            observation = {
                'global_in': global_in,
                'local_in': local_in,
                'local_in_two': local_in_two,
                'stats': stats
            }

        if self.with_local and self.with_stats and self.with_previous:
            action_vector = np.zeros(19)
            if action != None:
                action_vector[action] = 1

            observation = {
                # Global View
                'global_in': global_in_one_hot,
                # 'attr_global_1': global_in[:, :, 1],
                # 'attr_global_2': global_in[:, :, 2],
                # 'attr_global_3': global_in[:, :, 3],
                # 'attr_global_4': global_in[:, :, 4],
                # 'attr_global_5': global_in[:, :, 5],
                #'global_in_attributes': global_in[:, :, 1:]/5.,

                # Local View
                'local_in': local_in_one_hot,
                # 'attr_local_1': local_in[:, :, 1],
                # 'attr_local_2': local_in[:, :, 2],
                # 'attr_local_3': local_in[:, :, 3],
                # 'attr_local_4': local_in[:, :, 4],
                # 'attr_local_5': local_in[:, :, 5],
                #'local_in_attributes': local_in[:, :, 1:]/5.,

                # Local Two View
                'local_in_two': local_in_two_one_hot,
                # 'attr_local_two_1': local_in_two[:, :, 1],
                # 'attr_local_two_2': local_in_two[:, :, 2],
                # 'attr_local_two_3': local_in_two[:, :, 3],
                # 'attr_local_two_4': local_in_two[:, :, 4],
                # 'attr_local_two_5': local_in_two[:, :, 5],
                #'local_in_two_attributes': local_in_two[:, :, 1:],

                # Stats
                'agent_stats': stats[:16],
                'target_stats': stats[16:],
                'prev_action': action_vector
            }

        if self.with_local and self.with_stats and self.with_previous and self.with_class:
            action_vector = np.zeros(17)
            if action != None:
                action_vector[action] = 1

            observation = {
                'global_in': global_in,
                'local_in': local_in,
                'local_in_two': local_in_two,
                'stats': stats,
                'agent_class': agent_class,
                'enemy_class': enemy_class,
                'action': action_vector
            }

        if self.with_local and self.with_stats and self.with_previous and self.with_hp:
            action_vector = np.zeros(17)
            if action != None:
                action_vector[action] = 1

            observation = {
                'global_in': global_in,
                'local_in': local_in,
                'local_in_two': local_in_two,
                'hp': hp,
                'stats': stats,
                'prev_action': action_vector
            }

        return observation

    def print_observation(self, observation, actions = None, reward = None):
        try:
            # print(observation)
            # print(observation['global_in'])
            print('action = ' + str(actions))
            print('reward = ' + str(reward))
            sum = observation['global_in'][:,:,0]*0
            for i in range(1, 7):
                sum += observation['global_in'][:,:,i]*i
            sum = np.flip(np.transpose(sum), 0)
            print(sum)
            print(" ")
            print(observation['agent_stats'])
            print(observation['target_stats'])
            # print(observation['local_in'])
            # print(observation['local_in_two'])

        except Exception as e:
            pass


    def execute(self, actions, raw_obs=False):

        if self.manual_input:
            input_action = input('...')

            try:
                actions = input_action
            except ValueError:
                pass

        if isinstance(actions, str):
            actions = self.command_to_action(actions)
            print(actions)


        env_info = None
        #signal.alarm(0)
        while env_info == None:
            #signal.signal(signal.SIGALRM, self.handler)
            #signal.alarm(300000)
            try:
                if self.use_double_agent:
                    info = self.unity_env.step({self.default_brain : [actions], self.double_brain : []})
                    env_info = info[self.default_brain]
                else:
                    env_info = self.unity_env.step([actions])[self.default_brain]
            except Exception as exc:
                self.close()
                self.unity_env = self.open_unity_environment(self.game_name, self.no_graphics, seed = int(time.time()),
                                                             worker_id=self.worker_id)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
                print("The environment didn't respond, it was necessary to close and reopen it")

        if self.use_double_agent:
            while len(env_info.vector_observations) <= 0:
                double_info = info[self.double_brain]
                obs = self.get_input_observation(double_info)
                if np.random.rand() > self.play_with_random:
                    act = self.double_agent.eval([obs])[0]
                else:
                    act = np.random.randint(0, self.action_size)
                info = self.unity_env.step({self.default_brain: [], self.double_brain: [act]})
                env_info = info[self.default_brain]


        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        observation = self.get_input_observation(env_info, actions)

        self.count += 1

        # Update global timesteps
        self.global_timesteps += 1

        if self.verbose:
            self.print_observation(observation, reward=reward, actions=actions)

        if not raw_obs:
            return [observation, done, reward]
        else:
            return [env_info, done, reward]

    def command_to_action(self, command):

        switcher = {
            "w": 0,
            "d": 1,
            "x": 2,
            "a": 3,
            "e": 4,
            "c": 5,
            "z": 6,
            "q": 7,
            "s": 8,
            "y": 9,
            "j": 10,
            "n": 11,
            "g": 12,
            "u": 13,
            "m": 14,
            "b": 15,
            "t": 16,
            "f": 17,
            "r": 18
        }

        return switcher.get(command, 99)

    def set_config(self, config):
        self.config = config

    def handler(self, signum, frame):
        print("Timeout!")
        raise Exception("end of time")

    def reset(self, raw_obs=False):

        self.count = 0

        env_info = None

        while env_info == None:
            #signal.signal(signal.SIGALRM, self.handler)
            #signal.alarm(60)
            try:
                logging.getLogger("mlagents.envs").setLevel(logging.WARNING)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
            except Exception as exc:
                self.close()
                self.unity_env = self.open_unity_environment(self.game_name, self.no_graphics, seed=int(time.time()),
                                                             worker_id=self.worker_id)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
                print("The environment didn't respond, it was necessary to close and reopen it")

        if self.use_double_agent:
            while len(env_info.vector_observations) <= 0:
                env_info = self.unity_env.step()[self.default_brain]

        observation = self.get_input_observation(env_info)

        if self.verbose:
            self.print_observation(observation)

        if not raw_obs:
            return observation
        else:
            return env_info

    def close(self):
        self.unity_env.close()

    def open_unity_environment(self, game_name, no_graphics, seed, worker_id):
        return UnityEnvironment(game_name, no_graphics=no_graphics, seed=seed, worker_id=worker_id)

    def add_probs(self, probs):
        self.probabilities.append(probs[0])

    def get_last_entropy(self):
        entropy = 0
        for prob in self.probabilities[-1]:
            entropy += (prob + eps)*(math.log(prob + eps) + eps)

        return -entropy

    def entropy(self, probs):
        entropy = 0
        for prob in probs:
            entropy += (prob + eps) * (math.log(prob + eps) + eps)

        return -entropy

    def states(self):
        return {
            # Categorical values: 2 agent, 1 empty tile, 0 obstacle tile
            # 'global_in': {'shape': (10, 10, 52), 'type': 'float'},
            # 'local_in': {'shape': (5, 5, 52), 'type': 'float'},
            # 'local_in_two': {'shape': (3, 3, 52), 'type': 'float'},
            'global_in': {'shape': (10, 10, 52), 'type': 'float'},
            'local_in': {'shape': (5, 5, 52), 'type': 'float'},
            'local_in_two': {'shape': (3, 3, 52), 'type': 'float'},

            # Transformer
            # I have a max of 20 items, 1 agent and 1 target. Each entity is embedded to have dimension 256.
            # All entities are concatenated together to form a 22(number of entities)x256(number of features)
            # matrix, this will be passed to the Transformer layer.
            # The output of the transformer will alwyas be a 22x256 matrix, and this will be average pooled along the
            # feature axis to form a 256 feature vector.
            # 'items': {'shape': (20, 50), 'type': 'float'},
            # 'agent': {'shape': (1, 85), 'type': 'float'},
            # 'target': {'shape': (1, 8), 'type': 'float'},

            # 'agent_stats': {'shape': (14), 'num_values': 128, 'type': 'int'},
            # 'target_stats': {'shape': (4), 'num_values': 32, 'type': 'int'},

            'agent_stats': {'shape': (16), 'num_values': 129, 'type': 'int'},
            'target_stats': {'shape': (15), 'num_values': 125, 'type': 'int'},

            'prev_action': {'shape': (19), 'type': 'float'}
        }

    def actions(self):
        return {
                    'type': 'int',
                    'num_values': 19
                }

    def max_episode_timesteps(self):
        return self._max_episode_timesteps


class Info():
    def __init__(self, string):
        self.item = string

    def items(self):

        return self.item, self.item





