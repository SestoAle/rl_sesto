from layers.layers import *


# Define the input specification
def input_spec():
    global_cell_view = tf.compat.v1.placeholder(tf.int32, [None, 19, 19], name='global_cell_view')
    cell_view = tf.compat.v1.placeholder(tf.int32, [None, 7, 7], name='cell_view')
    in_range = tf.compat.v1.placeholder(tf.float32, [None, 1], name='in_range')
    actual_potion = tf.compat.v1.placeholder(tf.float32, [None, 1], name='actual_potion')
    agent_actual_HP = tf.compat.v1.placeholder(tf.float32, [None, 1], name='agent_actual_HP')
    target_actual_HP = tf.compat.v1.placeholder(tf.float32, [None, 1], name='target_actual_HP')

    return [global_cell_view, cell_view, in_range, actual_potion, agent_actual_HP, target_actual_HP]


# Change the observation in real states
def obs_to_state(obs):
    global_cell_view_batch = np.stack([np.asarray(state['global_cell_view']) for state in obs])
    cell_view_batch = np.stack([np.asarray(state['cell_view']) for state in obs])
    in_range_batch = np.stack([np.asarray(state['in_range']) for state in obs])
    actual_potion_batch = np.stack([np.asarray(state['actual_potion']) for state in obs])
    agent_actual_HP_batch = np.stack([np.asarray(state['agent_actual_HP']) for state in obs])
    target_actual_HP_batch = np.stack([np.asarray(state['target_actual_HP']) for state in obs])

    return [global_cell_view_batch, cell_view_batch, in_range_batch, actual_potion_batch, agent_actual_HP_batch,
            target_actual_HP_batch]


# Main network specification. Usually, this network will be followed by 2 FC layers
def network_spec(states, baseline=False):

    # Global cell view
    global_cell_view = tf.cast(states[0], tf.int32)

    emb_global = embedding(global_cell_view, indices=4, size=16)
    conv_global_1 = conv_layer_2d(emb_global, 16, [3, 3], name='conv_global_1', activation=tf.nn.relu)
    conv_global_2 = conv_layer_2d(conv_global_1, 32, [3, 3], name='conv_global_2', activation=tf.nn.relu)
    flat_global = tf.reshape(conv_global_2, [-1, 19 * 19 * 32])

    # Local cell view
    local_cell_view = tf.cast(states[1], tf.int32)

    emb_local = embedding(local_cell_view, indices=4, size=16)
    conv_local_1 = conv_layer_2d(emb_local, 16, [3, 3], name='conv_local_1', activation=tf.nn.relu)
    conv_local_2 = conv_layer_2d(conv_local_1, 32, [3, 3], name='conv_local_2', activation=tf.nn.relu)
    flat_local = tf.reshape(conv_local_2, [-1, 7 * 7 * 32])

    # Stato agenti
    state = tf.concat([states[2], states[3], states[4], states[5]], axis=1)
    fc_state = linear(state, 256, name='fc_gs', activation=tf.nn.relu)

    all_flat = tf.concat([flat_global, flat_local, fc_state], axis=1)

    return all_flat
