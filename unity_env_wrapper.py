from mlagents.envs import UnityEnvironment

import numpy as np

import time
import logging


class UnityEnvWrapper():

    def __init__(self, game_name=None, no_graphics=True, seed=None, worker_id=0, config=None, _max_episode_timesteps=40):

        self.game_name = game_name
        self.no_graphics = no_graphics
        self.seed = seed
        self.worker_id = worker_id
        self.unity_env = self.open_unity_environment(game_name, no_graphics, seed, worker_id)
        self.default_brain = self.unity_env.brain_names[0]

        self._max_episode_timesteps = _max_episode_timesteps

        self.set_config(config)

    def entropy(self, props):
        entr = 0
        for p in props:
            entr += p * np.log(p)

        return - entr

    def reset(self):

        env_info = None

        while env_info == None:
            try:
                logging.getLogger("mlagents.envs").setLevel(logging.WARNING)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]

            except Exception as exc:
                self.close()
                self.unity_env = self.open_unity_environment(self.game_name, self.no_graphics, seed=int(time.time()),
                                                             worker_id=self.worker_id)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
                print("The environment didn't respond, it was necessary to close and reopen it")

        obs = self.get_input_observation(env_info)

        return obs

    def execute(self, actions):

        env_info = None

        while env_info == None:
            try:
                env_info = self.unity_env.step([actions])[self.default_brain]

            except Exception as exc:
                self.close()
                self.unity_env = self.open_unity_environment(self.game_name, self.no_graphics, seed=int(time.time()),
                                                             worker_id=self.worker_id)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
                print("The environment didn't respond, it was necessary to close and reopen it")

        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        observation = self.get_input_observation(env_info)

        return [observation, done, reward]

    def open_unity_environment(self, game_name, no_graphics, seed, worker_id):
        return UnityEnvironment(game_name, no_graphics=no_graphics, seed=seed, worker_id=worker_id)

    def close(self):
        self.unity_env.close()

    def set_config(self, config):
        self.config = config

    def handler(self, signum, frame):
        print("Timeout!")
        raise Exception("end of time")

    def get_input_observation(self, env_info):

        '''
        observation = {
            'position': np.asarray(env_info.vector_observations[0][:2]),
            'forward_direction': np.asarray(env_info.vector_observations[0][2:3]),
            'target_position': np.asarray(env_info.vector_observations[0][3:5]),
            'env_objects': np.asarray(env_info.vector_observations[0][5:57]),
            'in_range': np.asarray(env_info.vector_observations[0][57:58]),
            'actual_potion': np.asarray(env_info.vector_observations[0][58:59])
        }
        '''

        observation = {
            'position': np.asarray(env_info.vector_observations[0][:2]),
            'forward_direction': np.asarray(env_info.vector_observations[0][2:3]),
            'target_position': np.asarray(env_info.vector_observations[0][3:5]),
            'cell_view': np.reshape(np.asarray(env_info.vector_observations[0][5:30], dtype=np.int32), (5, 5, 1)),
            'in_range': np.asarray(env_info.vector_observations[0][30:31]),
            'actual_potion': np.asarray(env_info.vector_observations[0][31:32])
        }

        return observation
