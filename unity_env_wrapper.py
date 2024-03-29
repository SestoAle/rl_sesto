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
            # Whether is a recurrent agent or not. Useful for self-play.
            self.recurrent = self.double_agent.recurrent
            if self.recurrent:
                self.double_internal = None

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
            # Reset internal state of the enemy
            if self.recurrent:
                self.double_internal = (np.zeros([1, self.double_agent.recurrent_size]),
                                   np.zeros([1, self.double_agent.recurrent_size]))
            self.prev_double_action = np.zeros(self.double_agent.action_size)
            # Può succedere che dopo un reset gli agenti non siano sincronizzati. Perchè?
            # Pensandoci all'inizio entrambi gli agenti devono chiedere l'azione
            # a meno che l'azione nell'episodio precedente non continui nell'episodio successivo
            while len(env_info.vector_observations) == 0:
                # Scegliere un'azione dalla rete secondo info[self.double_brain].vector_observations
                double_info = info[self.double_brain]
                if not self.recurrent:
                    double_obs = self.get_input_observation(double_info)
                    self.double_action = self.double_agent.eval_max([double_obs])[0]
                else:
                    double_reward = double_info.rewards[0]
                    double_obs = self.get_input_observation_with_action(double_info, self.prev_double_action, double_reward)
                    self.double_action, self.double_internal = self.double_agent.eval_max([double_obs],
                                                                                          self.double_internal)
                    self.double_action = self.double_action[0]
                    self.prev_double_action = self.double_action
                info = self.unity_env.step({self.default_brain: [], self.double_brain: self.double_action})
                env_info = info[self.default_brain]

            if len(info[self.double_brain].vector_observations) > 0:
                # Scegliere un'azione dalla rete secondo info[self.double_brain].vector_observations
                double_info = info[self.double_brain]
                if not self.recurrent:
                    double_obs = self.get_input_observation(double_info)
                    self.double_action = self.double_agent.eval_max([double_obs])[0]
                else:
                    double_reward = double_info.rewards[0]
                    double_obs = self.get_input_observation_with_action(double_info, self.prev_double_action,
                                                                        double_reward)
                    self.double_action, self.double_internal = self.double_agent.eval_max([double_obs],
                                                                                          self.double_internal)
                    self.double_action = self.double_action[0]
                    self.prev_double_action = self.double_action
            else:
                self.double_action = []

        if not self.recurrent:
            obs = self.get_input_observation(env_info)
        else:
            prev_action = np.zeros(self.double_agent.action_size)
            prev_reward = env_info.rewards[0]
            obs = self.get_input_observation_with_action(env_info, prev_action, prev_reward)
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
                if not self.recurrent:
                    double_obs = self.get_input_observation(double_info)
                    self.double_action = self.double_agent.eval_max([double_obs])[0]
                else:
                    double_reward = double_info.rewards[0]
                    double_obs = self.get_input_observation_with_action(double_info, self.prev_double_action,
                                                                        double_reward)
                    self.double_action, self.double_internal = self.double_agent.eval_max([double_obs],
                                                                                          self.double_internal)
                    self.double_action = self.double_action[0]
                    self.prev_double_action = self.double_action
                info = self.unity_env.step({self.default_brain: [], self.double_brain: self.double_action})
                env_info = info[self.default_brain]

            # Se il secondo agente non ha osservazioni, allora non vuole nessuna azione
            if len(info[self.double_brain].vector_observations) > 0:
                # Scegliere un'azione dalla rete secondo info[self.double_brain].vector_observations
                double_info = info[self.double_brain]
                if not self.recurrent:
                    double_obs = self.get_input_observation(double_info)
                    self.double_action = self.double_agent.eval_max([double_obs])[0]
                else:
                    double_reward = double_info.rewards[0]
                    double_obs = self.get_input_observation_with_action(double_info, self.prev_double_action,
                                                                        double_reward)
                    self.double_action, self.double_internal = self.double_agent.eval_max([double_obs],
                                                                                          self.double_internal)
                    self.double_action = self.double_action[0]
                    self.prev_double_action = self.double_action
            else:
                self.double_action = []

        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        if not self.recurrent:
            observation = self.get_input_observation(env_info)
        else:
            prev_action = actions
            prev_reward = reward
            observation = self.get_input_observation_with_action(env_info, prev_action, prev_reward)

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
        observation = {
            'position': np.asarray(env_info.vector_observations[0][:2]),
            'forward_direction': np.asarray(env_info.vector_observations[0][2:3]),
            'target_position': np.asarray(env_info.vector_observations[0][3:5]),
            'differences': np.asarray(env_info.vector_observations[0][5:7]),
            # Per passarlo all'embedding deve avere 3 dimensioni, quindi (batch_size, 5, 5) e non (batch_size, 5, 5, 1)
            'cell_view': np.reshape(np.asarray(env_info.vector_observations[0][7:56], dtype=np.int32), (7, 7)),
            'in_range': np.asarray(env_info.vector_observations[0][56:57]),
            'actual_potion': np.asarray(env_info.vector_observations[0][57:58]),
            'agent_actual_HP': np.asarray(env_info.vector_observations[0][58:59]),
            'target_actual_HP': np.asarray(env_info.vector_observations[0][59:60])
        }
        '''

        observation = {
            'target_transformer_input': np.reshape(np.asarray(env_info.vector_observations[0][:8], dtype=np.float32),
                                                   (1, 8)),
            'items_transformer_input': np.reshape(np.asarray(env_info.vector_observations[0][8:80], dtype=np.float32),
                                                  (9, 8)),
            'cell_view': np.reshape(np.asarray(env_info.vector_observations[0][80:129], dtype=np.int32), (7, 7)),
            'position': np.asarray(env_info.vector_observations[0][129:131]),
            'forward_direction': np.asarray(env_info.vector_observations[0][131:132]),
            'in_range': np.asarray(env_info.vector_observations[0][132:133], dtype=np.int32),
            'actual_health_potion': np.asarray(env_info.vector_observations[0][133:134], dtype=np.int32),
            'actual_bonus_potion': np.asarray(env_info.vector_observations[0][134:135], dtype=np.int32),
            'active_bonus_potion': np.asarray(env_info.vector_observations[0][135:136], dtype=np.int32),
            'agent_actual_HP': np.asarray(env_info.vector_observations[0][136:137], dtype=np.int32),
            'target_actual_HP': np.asarray(env_info.vector_observations[0][137:138], dtype=np.int32),
            'agent_actual_damage': np.asarray(env_info.vector_observations[0][138:139], dtype=np.int32),
            'target_actual_damage': np.asarray(env_info.vector_observations[0][139:140], dtype=np.int32),
            'agent_actual_def': np.asarray(env_info.vector_observations[0][140:141], dtype=np.int32),
            'target_actual_def': np.asarray(env_info.vector_observations[0][141:142], dtype=np.int32),
        }

        '''
        observation = {
            'global_cell_view': np.reshape(np.asarray(env_info.vector_observations[0][0:361], dtype=np.int32),
                                           (19, 19)),
            'cell_view': np.reshape(np.asarray(env_info.vector_observations[0][361:410], dtype=np.int32), (7, 7)),
            'in_range': np.asarray(env_info.vector_observations[0][410:411]),
            'actual_potion': np.asarray(env_info.vector_observations[0][411:412]),
            'agent_actual_HP': np.asarray(env_info.vector_observations[0][412:413]),
            'target_actual_HP': np.asarray(env_info.vector_observations[0][413:414]),
            'agent_actual_damage': np.asarray(env_info.vector_observations[0][414:415]),
            'target_actual_damage': np.asarray(env_info.vector_observations[0][415:416]),
            'agent_actual_def': np.asarray(env_info.vector_observations[0][416:417]),
            'target_actual_def': np.asarray(env_info.vector_observations[0][417:418]),
            'forward_direction': np.asarray(env_info.vector_observations[0][418:419])
        }
        '''

        return observation

    # For recurrent, we must add to input observation the previous action and reward
    def get_input_observation_with_action(self, env_info, action, reward):
        observation = self.get_input_observation(env_info)
        observation['prev_action'] = action
        observation['prev_reward'] = [reward]
        return observation
