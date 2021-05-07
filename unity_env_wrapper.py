from mlagents.envs import UnityEnvironment

import numpy as np

import time
import logging


class UnityEnvWrapper():

    def __init__(self, game_name=None, no_graphics=True, seed=None, worker_id=0, config=None, _max_episode_timesteps=40,
                 # Double agent
                 use_double_agent=False, double_agent=None
                 ):

        self.game_name = game_name
        self.no_graphics = no_graphics
        self.seed = seed
        self.worker_id = worker_id
        self.unity_env = self.open_unity_environment(game_name, no_graphics, seed, worker_id)
        self.default_brain = self.unity_env.brain_names[0]

        # Adversarial Play
        self.use_double_agent = use_double_agent
        if self.use_double_agent:
            self.double_brain = self.unity_env.brain_names[1]
            self.double_agent = double_agent

        self._max_episode_timesteps = _max_episode_timesteps

        self.set_config(config)

        self.count_moves = 0

    def entropy(self, props):
        entr = 0
        for p in props:
            entr += p * np.log(p)

        return - entr
    
    def reset(self):

        env_info = None

        while env_info is None:
            try:
                logging.getLogger("mlagents.envs").setLevel(logging.WARNING)

                info = self.unity_env.reset(train_mode=True, config=self.config)
                env_info = info[self.default_brain]

            except Exception as exc:
                self.close()
                self.unity_env = self.open_unity_environment(self.game_name, self.no_graphics, seed=int(time.time()),
                                                             worker_id=self.worker_id)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
                print("The environment didn't respond, it was necessary to close and reopen it")
        
        if self.use_double_agent: 
            # Può succedere che dopo un reset gli agenti non siano sincronizzati. Perchè?
            # Pensandoci all'inizio entrambi gli agenti devono chiedere l'azione
            # a meno che l'azione nell'episodio precedente non continui nell'episodio successivo
            while len(env_info.vector_observations) == 0:
                # Scegliere un'azione dalla rete secondo info[self.double_brain].vector_observations
                double_info = info[self.double_brain]
                double_obs = self.get_input_observation(double_info)
                self.double_action = self.double_agent.eval_max([double_obs])[0]
                info = self.unity_env.step({self.default_brain: [], self.double_brain: self.double_action})
                env_info = info[self.default_brain]

            if len(info[self.double_brain].vector_observations) > 0:
                # Scegliere un'azione dalla rete secondo info[self.double_brain].vector_observations
                double_info = info[self.double_brain]
                double_obs = self.get_input_observation(double_info)
                self.double_action = self.double_agent.eval_max([double_obs])[0]
            else:
                self.double_action = []

        obs = self.get_input_observation(env_info)

        return obs

    def execute(self, actions):

        env_info = None
        info = []

        while env_info is None:
            if self.use_double_agent:
                info = self.unity_env.step({self.default_brain: [actions], self.double_brain: self.double_action})

                env_info = info[self.default_brain]
            else:
                env_info = self.unity_env.step([actions])[self.default_brain]
        
        if self.use_double_agent:
            # Se il primo agente non ha osservazioni, allora è solo il secondo agente che vuole le azioni 
            while len(env_info.vector_observations) == 0:
                # Scegliere un'azione dalla rete secondo info[self.double_brain].vector_observations
                double_info = info[self.double_brain]
                double_obs = self.get_input_observation(double_info)
                self.double_action = self.double_agent.eval_max([double_obs])[0]
                info = self.unity_env.step({self.default_brain: [], self.double_brain: self.double_action})
                env_info = info[self.default_brain]
            
            # Se il secondo agente non ha osservazioni, allora non vuole nessuna azione
            if len(info[self.double_brain].vector_observations) > 0:
                # Scegliere un'azione dalla rete secondo info[self.double_brain].vector_observations
                double_info = info[self.double_brain]
                double_obs = self.get_input_observation(double_info)
                self.double_action = self.double_agent.eval_max([double_obs])[0]
            else:
                self.double_action = []

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
            'rays': np.reshape(np.asarray(env_info.vector_observations[0][5:185]), (36, 5)),
            'in_range': np.asarray(env_info.vector_observations[0][185:186]),
            'actual_potion': np.asarray(env_info.vector_observations[0][186:187])
        }
        '''
        observation = {
            'position': np.asarray(env_info.vector_observations[0][:2]),
            'forward_direction': np.asarray(env_info.vector_observations[0][2:3]),
            'target_position': np.asarray(env_info.vector_observations[0][3:5]),
            # Per passarlo all'embedding deve avere 3 dimensioni, quindi (batch_size, 5, 5) e non (batch_size, 5, 5, 1)
            'cell_view': np.reshape(np.asarray(env_info.vector_observations[0][5:30], dtype=np.int32), (5, 5)),
            'in_range': np.asarray(env_info.vector_observations[0][30:31]),
            'actual_potion': np.asarray(env_info.vector_observations[0][31:32]),
            'actual_HP': np.asarray(env_info.vector_observations[0][32:33])
        }

        return observation
