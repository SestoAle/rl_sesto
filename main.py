from agents.PPO import PPO
from runner.runner import Runner
from architectures.transformer_arch_stats_emb import *
from runner.parallel_runner import Runner as ParallelRunner
import os
import time
import tensorflow as tf
from unity_env_wrapper import UnityEnvWrapper
import argparse

from reward_model.reward_model import RewardModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='model')
parser.add_argument('-gn', '--game-name', help="The name of the game", default=None)
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-sf', '--save-frequency', help="How many episodes after save the model", default=25000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=40)
parser.add_argument('-se', '--sampled-env', help="IRL", default=20)
parser.add_argument('-rc', '--recurrent', dest='recurrent', action='store_true')
parser.add_argument('-pl', '--parallel', dest='parallel', action='store_true')
parser.add_argument('-ev', '--eval', dest='evaluation', action='store_true', help="If evaluation")

# Parse argument for adversarial-play
parser.add_argument('-ad', '--adversarial-play', help="Whether to use adversarial play",
                    dest='adversarial_play', action='store_true')
parser.add_argument('-sd', '--sample-adversarial', help="Whether to use sample the adversarial net",
                    dest='sample_adversarial', action='store_true')
parser.add_argument('-af', '--adversarial-frequency', help="How many episodes after save the enemy", default=10000)
parser.set_defaults(adversarial_play=False)
parser.set_defaults(sample_adversarial=False)

# Parse arguments for Inverse Reinforcement Learning
parser.add_argument('-irl', '--inverse-reinforcement-learning', dest='use_reward_model', action='store_true')
parser.add_argument('-rf', '--reward-frequency', help="How many episode before update the reward model", default=15)
parser.add_argument('-rm', '--reward-model', help="The name of the reward model", default='warrior_10')
parser.add_argument('-dn', '--dems-name', help="The name of the demonstrations file", default='dems_archer.pkl')
parser.add_argument('-fr', '--fixed-reward-model', help="Whether to use a trained reward model",
                    dest='fixed_reward_model', action='store_true')

parser.set_defaults(use_reward_model=False)
parser.set_defaults(fixed_reward_model=False)
parser.set_defaults(recurrent=False)
parser.set_defaults(parallel=False)
parser.set_defaults(evaluation=False)

args = parser.parse_args()

if __name__ == "__main__":

    # Game arguments
    game_name = args.game_name
    model_name = args.model_name
    work_id = int(args.work_id)
    save_frequency = int(args.save_frequency)
    logging = int(args.logging)
    # max_episode_timestep = int(args.max_timesteps)
    sampled_env = int(args.sampled_env)

    evaluation = args.evaluation

    # Whether to use parallel executions
    parallel = args.parallel
    n_envs = 10

    # Adversarial play
    adversarial_play = args.adversarial_play
    sample_adversarial = args.sample_adversarial
    adversarial_frequency = int(args.adversarial_frequency)

    # IRL
    use_reward_model = args.use_reward_model
    reward_model_name = args.reward_model
    fixed_reward_model = args.fixed_reward_model
    dems_name = args.dems_name
    reward_frequency = int(args.reward_frequency)

    max_episode_timestep = 100

    curriculum = {
        'current_step': 0,
        "thresholds": [50000, 50000, 50000, 50000, 50000, 175000],
        "parameters": {
            "spawn_range": [10, 11, 12, 13, 14, 14, 14],
            "attack_range_epsilon": [1, 1, 1, 1, 1, 1, 1],
            "agent_update_rate": [10, 10, 10, 10, 10, 10, 10],
            "target_update_rate": [10, 10, 10, 10, 10, 10, 10],
            "target_speed": [4, 5, 6, 6, 6, 6, 6],
            "min_agent_HP": [1, 1, 5, 5, 5, 20, 20],
            "max_agent_HP": [1, 5, 15, 20, 20, 20, 20],
            "min_target_HP": [1, 1, 5, 5, 5, 20, 20],
            "max_target_HP": [5, 5, 10, 15, 20, 20, 20],
            "max_items": [9, 9, 9, 8, 7, 6, 6],
            "health_potion_frequency": [0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5],
            "shield_frequency": [0.03, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15],
            "sword_frequency": [0.03, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15],
            "bonus_potion_frequency": [0.04, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
        }
    }

    # Total episode of training
    total_episode = 400100
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
        agent = PPO(sess, input_spec=input_spec, network_spec=network_spec, obs_to_state=obs_to_state,
                    p_lr=7e-6, p_num_itr=10, v_lr=7e-5, v_batch_fraction=1.0, v_num_itr=1, action_size=4,
                    action_type='continuous', distribution='beta',
                    memory=memory, model_name=model_name, recurrent=args.recurrent, frequency_mode=frequency_mode)
        # Initialize variables of models
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    double_agent = None
    # Create double agent
    if adversarial_play:
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            double_sess = tf.compat.v1.Session(graph=graph)
            double_agent = PPO(double_sess, input_spec=input_spec, network_spec=network_spec, obs_to_state=obs_to_state,
                    p_lr=7e-6, p_num_itr=10, v_lr=7e-5, v_batch_fraction=1.0, v_num_itr=1, action_size=4,
                    action_type='continuous', distribution='beta',
                    memory=memory, model_name=model_name, recurrent=args.recurrent, frequency_mode=frequency_mode)
            # Initialize variables of models
            init = tf.compat.v1.global_variables_initializer()
            double_sess.run(init)

    # Open the environment with all the desired flags
    if not parallel:
        env = UnityEnvWrapper(game_name, no_graphics=True, seed=None, worker_id=work_id,
                              _max_episode_timesteps=max_episode_timestep,
                              use_double_agent=adversarial_play, double_agent=double_agent)
    else:
        # If parallel, create more environemnts
        envs = []
        for i in range(1, n_envs + 1):
            envs.append(UnityEnvWrapper(game_name, no_graphics=True, seed=i, worker_id=work_id + i,
                                        _max_episode_timesteps=max_episode_timestep,
                                        use_double_agent=adversarial_play, double_agent=double_agent))

    # If we want to use IRL, create a reward model
    reward_model = None
    if use_reward_model:
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            reward_sess = tf.compat.v1.Session(graph=graph)
            reward_model = RewardModel(actions_size=4, policy=agent, sess=reward_sess, name=model_name)
            # Initialize variables of models
            init = tf.compat.v1.global_variables_initializer()
            reward_sess.run(init)
            # If we want, we can use an already trained reward model
            if fixed_reward_model:
                reward_model.load_model(reward_model_name)
                print("Model loaded!")

    # Create runner
    if not parallel:
        runner = Runner(agent=agent, frequency=frequency, env=env, save_frequency=save_frequency, logging=logging,
                        total_episode=total_episode,
                        curriculum=curriculum,
                        frequency_mode=frequency_mode,
                        reward_model=reward_model,
                        reward_frequency=reward_frequency,
                        dems_name=dems_name,
                        fixed_reward_model=fixed_reward_model,
                        curriculum_mode='episodes',
                        evaluation=evaluation,
                        # Adversarial play
                        double_agent=double_agent,
                        adversarial_play=adversarial_play,
                        adversarial_frequency=adversarial_frequency,
                        sample_adversarial=sample_adversarial)
    else:
        runner = ParallelRunner(agent=agent, frequency=frequency, envs=envs, save_frequency=save_frequency,
                                logging=logging,
                                total_episode=total_episode,
                                curriculum=curriculum,
                                frequency_mode=frequency_mode,
                                reward_model=reward_model,
                                reward_frequency=reward_frequency,
                                dems_name=dems_name,
                                fixed_reward_model=fixed_reward_model,
                                curriculum_mode='episodes',
                                evaluation=False,
                                # Adversarial play
                                double_agent=double_agent,
                                adversarial_play=adversarial_play,
                                adversarial_frequency=adversarial_frequency,
                                sample_adversarial=sample_adversarial)
    try:
        runner.run()
    finally:
        if not parallel:
            env.close()
        else:
            for env in envs:
                env.close()
