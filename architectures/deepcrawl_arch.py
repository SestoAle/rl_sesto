from layers.layers import *

# Define the input specification of DeepCrawl
def input_spec():
    global_state = tf.compat.v1.placeholder(tf.float32, [None, 10, 10, 52], name='global_state')
    local_state = tf.compat.v1.placeholder(tf.float32, [None, 5, 5, 52], name='local_state')
    local_two_state = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 52], name='local_two_state')
    agent_stats = tf.compat.v1.placeholder(tf.int32, [None, 16], name='agent_stats')
    target_stats = tf.compat.v1.placeholder(tf.int32, [None, 15], name='target_stats')

    return [global_state, local_state, local_two_state, agent_stats, target_stats]

# Change the observation in real states
def obs_to_state(obs):
    global_batch = np.stack([np.asarray(state['global_in']) for state in obs])
    local_batch = np.stack([np.asarray(state['local_in']) for state in obs])
    local_two_batch = np.stack([np.asarray(state['local_in_two']) for state in obs])
    agent_stats_batch = np.stack([np.asarray(state['agent_stats']) for state in obs])
    target_stats_batch = np.stack([np.asarray(state['target_stats']) for state in obs])

    return [global_batch, local_batch, local_two_batch, agent_stats_batch, target_stats_batch]

# Main network specification. Usually, this network will be followed by 2 FC layers
def network_spec(states, baseline=False):

    global_state = states[0]
    local_state = states[1]
    local_two_state = states[2]
    agent_stats = states[3]
    target_stats = states[4]

    conv_10 = conv_layer_2d(global_state, 32, [1, 1], name='conv_10', activation=tf.nn.tanh, bias=False)
    conv_11 = conv_layer_2d(conv_10, 32, [3, 3], name='conv_11', activation=tf.nn.relu)
    conv_12 = conv_layer_2d(conv_11, 64, [3, 3], name='conv_12', activation=tf.nn.relu)
    flat_11 = tf.reshape(conv_12, [-1, 10 * 10 * 64])

    conv_20 = conv_layer_2d(local_state, 32, [1, 1], name='conv_20', activation=tf.nn.tanh, bias=False)
    conv_21 = conv_layer_2d(conv_20, 32, [3, 3], name='conv_21', activation=tf.nn.relu)
    conv_22 = conv_layer_2d(conv_21, 64, [3, 3], name='conv_22', activation=tf.nn.relu)
    flat_21 = tf.reshape(conv_22, [-1, 5 * 5 * 64])

    conv_30 = conv_layer_2d(local_two_state, 32, [1, 1], name='conv_30', activation=tf.nn.tanh, bias=False)
    conv_31 = conv_layer_2d(conv_30, 32, [3, 3], name='conv_31', activation=tf.nn.relu)
    conv_32 = conv_layer_2d(conv_31, 64, [3, 3], name='conv_32', activation=tf.nn.relu)
    flat_31 = tf.reshape(conv_32, [-1, 3 * 3 * 64])

    embs_41 = embedding(agent_stats, 129, 256, name='embs_41')
    embs_41 = tf.reshape(embs_41, [-1, 16 * 256])
    if not baseline:
        flat_41 = linear(embs_41, 256, name='fc_41', activation=tf.nn.relu)
    else:
        flat_41 = linear(embs_41, 128, name='fc_41', activation=tf.nn.relu)

    embs_51 = embedding(target_stats, 125, 256, name='embs_51')
    embs_51 = tf.reshape(embs_51, [-1, 15 * 256])
    if not baseline:
        flat_51 = linear(embs_51, 256, name='fc_51', activation=tf.nn.relu)
    else:
        flat_51 = linear(embs_51, 128, name='fc_51', activation=tf.nn.relu)

    all_flat = tf.concat([flat_11, flat_21, flat_31, flat_41, flat_51], axis=1)

    return all_flat
