from agents.PPO import PPO
from runner.runner import Runner
import os
import time
import tensorflow as tf
from unity_env_wrapper import UnityEnvWrapper
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='warrior')
parser.add_argument('-gn', '--game-name', help="The name of the game", default='envs/DeepCrawl-Procedural-4')
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)

args = parser.parse_args()

if __name__ == "__main__":

    # DeepCrawl arguments
    game_name = args.game_name
    work_id = int(args.work_id)
    model_name = args.model_name
    use_reward_model = args.use_reward_model
    reward_model_name = args.reward_model

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = {
        'current_step': 0,
        'thresholds': [1e6, 0.8e6, 1e6, 1e6],
        'parameters':
            {
                'minTargetHp': [1, 10, 10, 10, 10],
                'maxTargetHp': [1, 10, 20, 20, 20],
                'minAgentHp': [15, 10, 5, 5, 10],
                'maxAgentHp': [20, 20, 20, 20, 20],
                'minNumLoot': [0.2, 0.2, 0.2, 0.08, 0.04],
                'maxNumLoot': [0.2, 0.2, 0.2, 0.3, 0.3],
                'minAgentMp': [0, 0, 0, 0, 0],
                'maxAgentMp': [0, 0, 0, 0, 0],
                'numActions': [17, 17, 17, 17, 17],
                # Agent statistics
                'agentAtk': [4, 4, 4, 4, 4],
                'agentDef': [3, 3, 3, 3, 3],
                'agentDes': [0, 0, 0, 0, 0],

                'minStartingInitiative': [1, 1, 1, 1, 1],
                'maxStartingInitiative': [1, 1, 1, 1, 1]
            }
    }

    # Total episode of training
    total_episode = 1e10
    # Frequency of training (in episode)
    frequency = 5
    # Memory of the agent (in episode)
    memory = 10
    # Frequency of logging
    logging = 100
    # Frequency of saving
    save_frequency = 3000
    # Max timestep for episode
    max_episode_timestep = 100

    # Open the environment with all the desired flags
    env = UnityEnvWrapper(game_name, no_graphics=True, seed=int(time.time()),
                                  worker_id=work_id, with_stats=True, size_stats=31,
                                  size_global=10, agent_separate=False, with_class=False, with_hp=False,
                                  with_previous=True, verbose=False, manual_input=False,
                                  _max_episode_timesteps=max_episode_timestep)

    # Create agent
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session(graph=graph)
        agent = PPO(sess=sess, memory=memory, model_name=model_name)
        # Initialize variables of models
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    # Create runner
    runner = Runner(agent, frequency, save_frequency, logging, total_episode, curriculum)
    try:
        runner.run(env)
    finally:
        #save_model(history, model_name, curriculum, agent)
        env.close()
