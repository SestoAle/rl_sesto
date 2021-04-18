from mlagents.envs import UnityEnvironment
import numpy as np
from threading import Thread
import time
from agents.PPO_openworld import PPO
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class OpenWorldEnv:

    def __init__(self, game_name, no_graphics, worker_id):
        self.no_graphics = no_graphics
        self.unity_env = UnityEnvironment(game_name, no_graphics=no_graphics, seed=None, worker_id=worker_id)
        self._max_episode_timesteps = 600
        self.default_brain = self.unity_env.brain_names[0]
        self.config = None
        self.actions_eps = 0.1
        self.previous_action = [0, 0]


    def execute(self, actions):
        env_info = self.unity_env.step([actions])[self.default_brain]
        self.step += 1
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        state = dict(global_in=env_info.vector_observations[0])
        # Concatenate last previous action
        state['global_in'] = np.concatenate([state['global_in'], self.previous_action])

        if self.step > self._max_episode_timesteps:
            done = True

        return state, done, reward

    def reset(self):
        self.previous_action = [0, 0]
        env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
        state = dict(global_in=env_info.vector_observations[0])
        # Concatenate last previous action
        state['global_in'] = np.concatenate([state['global_in'], self.previous_action])
        self.step = 0
        return state

    def entropy(self, probs):
        entr = 0
        for p in probs:
            entr += (p * np.log(p))
        return -entr

    def set_config(self, config):
        return None

    def close(self):
        self.unity_env.close()

# Method for count time after each episode
def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

class MyThread(Thread):
    def __init__(self, env, buffer, agent, index):
        self.env = env
        self.buffer = buffer
        self.agent = agent
        self.index = index
        super().__init__()

    def run(self) -> None:
        done = False
        state = self.env.reset()
        while not done:
            actions = self.agent.eval([state])[0]
            state, done, reward = self.env.execute(actions)
            self.buffer[self.index].append(state)

class MySecondThread(Thread):
    def __init__(self, env, buffer, actions, index, dones):
        self.env = env
        self.buffer = buffer
        self.actions = actions
        self.index = index
        self.dones = dones
        super().__init__()

    def run(self) -> None:
        state, done, reward = self.env.execute(self.actions)
        self.buffer[self.index].append(state)
        self.dones.append(done)
        if done:
            self.env.reset()

if __name__ == '__main__':

    # Create agent
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session(graph=graph)
        agent = PPO(sess, action_type='continuous', action_size=2, model_name='openworl_prev', p_lr=5e-6, v_lr=5e-6,
                    recurrent=False)
        # Initialize variables of models
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    parallel = True
    vectorized = False
    start_time = time.time()
    states = 200
    total_episode = 10
    real_buffer = []
    if parallel:

        buffer = []
        envs = []
        num_thread = 5
        for i in range(num_thread):
            buffer.append([])
        for i in range(num_thread):
            envs.append(OpenWorldEnv(game_name='envs/OpenWorldLittle', no_graphics=True, worker_id=80+i))

        for i, e in enumerate(envs):
            buffer[i].append(e.reset())

        threads = []
        current_episode = 0

        while current_episode < total_episode:

            for i, e in enumerate(envs):
                x = MyThread(e, buffer, agent, i)
                threads.append(x)
                x.start()

            for t in threads:
                t.join()
                current_episode += 1

        for i in buffer:
            real_buffer = np.concatenate([real_buffer, i])

        print(len(real_buffer))
    elif vectorized:
        buffer = []
        envs = []
        num_thread = 5
        for i in range(num_thread):
            buffer.append([])
        for i in range(num_thread):
            envs.append(OpenWorldEnv(game_name='envs/OpenWorldLittle', no_graphics=True, worker_id=80 + i))

        batch = []
        dones = []
        for i, e in enumerate(envs):
            s = e.reset()
            buffer[i].append(s)
            batch.append(s)

        threads = []
        current_episode = 0

        while np.sum(dones) < total_episode:
            for i, e in enumerate(envs):
                action, logprob, oldprob = agent.eval(batch)
                x = MySecondThread(e, buffer, action[i], i, dones)
                threads.append(x)
                x.start()

            for t in threads:
                t.join()
                batch[t.index] = buffer[t.index][-1]

        for i in buffer:
            real_buffer = np.concatenate([real_buffer, i])

        print(len(real_buffer))
    else:
        num_thread = 1
        envs = []
        buffer = []
        for i in range(num_thread):
            envs.append(OpenWorldEnv(game_name='envs/OpenWorldLittle', no_graphics=True, worker_id=80+i))

        for e in envs:
            state = e.reset()
        current_episode = 0
        while current_episode < total_episode:
            for e in envs:
                actions = agent.eval([state])[0]
                state, done, reward = e.execute(actions)
                if done:
                    e.reset()
                    current_episode += 1
                buffer.append(state)
        print(len(buffer))
    timer(start_time, time.time())