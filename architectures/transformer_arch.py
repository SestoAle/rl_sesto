from layers.layers import *


# Define the input specification
def input_spec():

    # Global
    target_transformer_input = tf.compat.v1.placeholder(tf.float32, [None, 1, 8], name='target_transformer_input')
    items_transformer_input = tf.compat.v1.placeholder(tf.float32, [None, 9, 8], name='items_transformer_input')

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
    agent_actual_damage = tf.compat.v1.placeholder(tf.float32, [None, 1], name='agent_actual_damage')
    target_actual_damage = tf.compat.v1.placeholder(tf.float32, [None, 1], name='target_actual_damage')
    agent_actual_def = tf.compat.v1.placeholder(tf.float32, [None, 1], name='agent_actual_def')
    target_actual_def = tf.compat.v1.placeholder(tf.float32, [None, 1], name='target_actual_def')

    return [target_transformer_input, items_transformer_input, cell_view, position, forward_direction, in_range,
            actual_potion, agent_actual_HP, target_actual_HP, agent_actual_damage, target_actual_damage,
            agent_actual_def, target_actual_def]


# Change the observation in real states
def obs_to_state(obs):

    target_transformer_input_batch = np.stack([np.asarray(state['target_transformer_input']) for state in obs])
    items_transformer_input_batch = np.stack([np.asarray(state['items_transformer_input']) for state in obs])

    cell_view_batch = np.stack([np.asarray(state['cell_view']) for state in obs])

    position_batch = np.stack([np.asarray(state['position']) for state in obs])
    forward_direction_batch = np.stack([np.asarray(state['forward_direction']) for state in obs])
    in_range_batch = np.stack([np.asarray(state['in_range']) for state in obs])

    actual_potion_batch = np.stack([np.asarray(state['actual_potion']) for state in obs])
    agent_actual_HP_batch = np.stack([np.asarray(state['agent_actual_HP']) for state in obs])
    target_actual_HP_batch = np.stack([np.asarray(state['target_actual_HP']) for state in obs])
    agent_actual_damage_batch = np.stack([np.asarray(state['agent_actual_damage']) for state in obs])
    target_actual_damage_batch = np.stack([np.asarray(state['target_actual_damage']) for state in obs])
    agent_actual_def_batch = np.stack([np.asarray(state['agent_actual_def']) for state in obs])
    target_actual_def_batch = np.stack([np.asarray(state['target_actual_def']) for state in obs])

    return [target_transformer_input_batch, items_transformer_input_batch, cell_view_batch, position_batch,
            forward_direction_batch, in_range_batch, actual_potion_batch, agent_actual_HP_batch, target_actual_HP_batch,
            agent_actual_damage_batch, target_actual_damage_batch, agent_actual_def_batch, target_actual_def_batch]


# Main network specification. Usually, this network will be followed by 2 FC layers
def network_spec(states, baseline=False):

    target = states[0]
    target_mask = create_mask(target, 99)
    target = linear(target, 2048, name='target_entity_emb', activation=tf.nn.tanh)

    items = states[1]
    items_mask = create_mask(items, 99)
    items = linear(items, 2048, name='items_entity_emb', activation=tf.nn.tanh)

    entity_embeddings = tf.concat([target, items], axis=1)
    mask = tf.concat([target_mask, items_mask], axis=2)

    # Global Transformer
    global_transformer, _ = transformer(entity_embeddings, n_head=8, hidden_size=2048, mask_value=99,
                                        with_embeddings=False, name='global_transformer', mask=mask, pooling='max')
    flat_global = tf.reshape(global_transformer, [-1, 2048])

    # Local Cell View
    cell_view = tf.cast(states[2], tf.int32)

    emb = embedding(cell_view, indices=7, size=32)
    conv_21 = conv_layer_2d(emb, 32, [3, 3], name='conv_21', activation=tf.nn.relu)
    conv_22 = conv_layer_2d(conv_21, 64, [3, 3], name='conv_22', activation=tf.nn.relu)
    flat_local = tf.reshape(conv_22, [-1, 7 * 7 * 64])

    # Agent position + stats
    stats = tf.concat([states[3], states[4], states[5], states[6], states[7], states[8], states[9], states[10],
                       states[11], states[12]], axis=1)
    fc_stats = linear(stats, 1024, name='fc_stats', activation=tf.nn.relu)

    all_flat = tf.concat([flat_global, fc_stats, flat_local], axis=1)

    return all_flat
