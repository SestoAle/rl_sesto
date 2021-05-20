from layers.layers import *


# Define the input specification
def input_spec():

    # Global
    transformer_global_input = tf.compat.v1.placeholder(tf.float32, [None, 10, 6], name='transformer_global_input')

    # Local
    cell_view = tf.compat.v1.placeholder(tf.int32, [None, 7, 7], name='cell_view')

    # Agent Position
    position = tf.compat.v1.placeholder(tf.float32, [None, 2], name='position')
    forward_direction = tf.compat.v1.placeholder(tf.float32, [None, 1], name='forward_direction')
    in_range = tf.compat.v1.placeholder(tf.float32, [None, 1], name='in_range')

    # Stats
    actual_potion = tf.compat.v1.placeholder(tf.float32, [None, 1], name='actual_potion')
    agent_actual_HP = tf.compat.v1.placeholder(tf.float32, [None, 1], name='agent_actual_HP')
    target_actual_HP = tf.compat.v1.placeholder(tf.float32, [None, 1], name='target_actual_HP')

    return [transformer_global_input, cell_view, position, forward_direction, in_range, actual_potion,
            agent_actual_HP, target_actual_HP]


# Change the observation in real states
def obs_to_state(obs):

    transformer_global_input_batch = np.stack([np.asarray(state['transformer_global_input']) for state in obs])

    cell_view_batch = np.stack([np.asarray(state['cell_view']) for state in obs])

    position_batch = np.stack([np.asarray(state['position']) for state in obs])
    forward_direction_batch = np.stack([np.asarray(state['forward_direction']) for state in obs])
    in_range_batch = np.stack([np.asarray(state['in_range']) for state in obs])

    actual_potion_batch = np.stack([np.asarray(state['actual_potion']) for state in obs])
    agent_actual_HP_batch = np.stack([np.asarray(state['agent_actual_HP']) for state in obs])
    target_actual_HP_batch = np.stack([np.asarray(state['target_actual_HP']) for state in obs])

    return [transformer_global_input_batch, cell_view_batch, position_batch, forward_direction_batch, in_range_batch,
            actual_potion_batch, agent_actual_HP_batch, target_actual_HP_batch]


# Main network specification. Usually, this network will be followed by 2 FC layers
def network_spec(states, baseline=False):

    # Global Transformer
    global_transformer, _ = transformer(states[0], n_head=4, hidden_size=1024, mask_value=99, with_embeddings=True,
                                        name='global_transformer', pooling='max')
    flat_global = tf.reshape(global_transformer, [-1, 1024])

    # Local Cell View
    cell_view = tf.cast(states[1], tf.int32)

    emb = embedding(cell_view, indices=7, size=16)
    conv_21 = conv_layer_2d(emb, 16, [3, 3], name='conv_21', activation=tf.nn.relu)
    conv_22 = conv_layer_2d(conv_21, 32, [3, 3], name='conv_22', activation=tf.nn.relu)
    flat_local = tf.reshape(conv_22, [-1, 7 * 7 * 32])

    # Agent position + stats
    stats = tf.concat([states[2], states[3], states[4], states[5], states[6], states[7]], axis=1)
    fc_stats = linear(stats, 512, name='fc_gs', activation=tf.nn.relu)

    all_flat = tf.concat([flat_global, flat_local, fc_stats], axis=1)

    return all_flat
