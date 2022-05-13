from layers.layers import *

stats_embedding_size = 16

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
    in_range = tf.compat.v1.placeholder(tf.int32, [None, 1], name='in_range')

    # Potions
    actual_health_potion = tf.compat.v1.placeholder(tf.int32, [None, 1], name='actual_health_potion')
    actual_bonus_potion = tf.compat.v1.placeholder(tf.int32, [None, 1], name='actual_bonus_potion')
    active_bonus_potion = tf.compat.v1.placeholder(tf.int32, [None, 1], name='active_bonus_potion')

    # Stats
    agent_actual_HP = tf.compat.v1.placeholder(tf.int32, [None, 1], name='agent_actual_HP')
    target_actual_HP = tf.compat.v1.placeholder(tf.int32, [None, 1], name='target_actual_HP')
    agent_actual_damage = tf.compat.v1.placeholder(tf.int32, [None, 1], name='agent_actual_damage')
    target_actual_damage = tf.compat.v1.placeholder(tf.int32, [None, 1], name='target_actual_damage')
    agent_actual_def = tf.compat.v1.placeholder(tf.int32, [None, 1], name='agent_actual_def')
    target_actual_def = tf.compat.v1.placeholder(tf.int32, [None, 1], name='target_actual_def')

    return [target_transformer_input, items_transformer_input, cell_view, position, forward_direction, in_range,
            actual_health_potion, actual_bonus_potion, active_bonus_potion, agent_actual_HP, target_actual_HP,
            agent_actual_damage, target_actual_damage, agent_actual_def, target_actual_def]


# Change the observation in real states
def obs_to_state(obs):
    target_transformer_input_batch = np.stack([np.asarray(state['target_transformer_input']) for state in obs])
    items_transformer_input_batch = np.stack([np.asarray(state['items_transformer_input']) for state in obs])

    cell_view_batch = np.stack([np.asarray(state['cell_view']) for state in obs])

    position_batch = np.stack([np.asarray(state['position']) for state in obs])
    forward_direction_batch = np.stack([np.asarray(state['forward_direction']) for state in obs])
    in_range_batch = np.stack([np.asarray(state['in_range']) for state in obs])

    actual_health_potion_batch = np.stack([np.asarray(state['actual_health_potion']) for state in obs])
    actual_bonus_potion_batch = np.stack([np.asarray(state['actual_bonus_potion']) for state in obs])
    active_bonus_potion_batch = np.stack([np.asarray(state['active_bonus_potion']) for state in obs])

    agent_actual_HP_batch = np.stack([np.asarray(state['agent_actual_HP']) for state in obs])
    target_actual_HP_batch = np.stack([np.asarray(state['target_actual_HP']) for state in obs])
    agent_actual_damage_batch = np.stack([np.asarray(state['agent_actual_damage']) for state in obs])
    target_actual_damage_batch = np.stack([np.asarray(state['target_actual_damage']) for state in obs])
    agent_actual_def_batch = np.stack([np.asarray(state['agent_actual_def']) for state in obs])
    target_actual_def_batch = np.stack([np.asarray(state['target_actual_def']) for state in obs])

    return [target_transformer_input_batch, items_transformer_input_batch, cell_view_batch, position_batch,
            forward_direction_batch, in_range_batch, actual_health_potion_batch, actual_bonus_potion_batch,
            active_bonus_potion_batch, agent_actual_HP_batch, target_actual_HP_batch, agent_actual_damage_batch,
            target_actual_damage_batch, agent_actual_def_batch, target_actual_def_batch]


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
    cell_view = states[2]

    emb = embedding(cell_view, indices=9, size=32)
    conv_21 = conv_layer_2d(emb, 32, [3, 3], name='conv_21', activation=tf.nn.relu)
    conv_22 = conv_layer_2d(conv_21, 64, [3, 3], name='conv_22', activation=tf.nn.relu)
    flat_local = tf.reshape(conv_22, [-1, 7 * 7 * 64])

    # Agent position + hard embeddings for stats

    in_range = states[5]
    in_range = embedding(in_range, indices=2, size=stats_embedding_size)
    in_range = tf.reshape(in_range, [-1, stats_embedding_size])

    health_potion = states[6]
    health_potion = embedding(health_potion, indices=2, size=stats_embedding_size)
    health_potion = tf.reshape(health_potion, [-1, stats_embedding_size])

    bonus_potion = states[7]
    bonus_potion = embedding(bonus_potion, indices=2, size=stats_embedding_size)
    bonus_potion = tf.reshape(bonus_potion, [-1, stats_embedding_size])

    active_bonus_potion = states[8]
    active_bonus_potion = embedding(active_bonus_potion, indices=2, size=stats_embedding_size)
    active_bonus_potion = tf.reshape(active_bonus_potion, [-1, stats_embedding_size])

    agent_HP = states[9]
    agent_HP = embedding(agent_HP, indices=21, size=stats_embedding_size)
    agent_HP = tf.reshape(agent_HP, [-1, stats_embedding_size])

    target_HP = states[10]
    target_HP = embedding(target_HP, indices=21, size=stats_embedding_size)
    target_HP = tf.reshape(target_HP, [-1, stats_embedding_size])

    agent_damage = states[11]
    agent_damage = embedding(agent_damage, indices=20, size=stats_embedding_size)
    agent_damage = tf.reshape(agent_damage, [-1, stats_embedding_size])

    target_damage = states[12]
    target_damage = embedding(target_damage, indices=20, size=stats_embedding_size)
    target_damage = tf.reshape(target_damage, [-1, stats_embedding_size])

    agent_def = states[13]
    agent_def = embedding(agent_def, indices=15, size=stats_embedding_size)
    agent_def = tf.reshape(agent_def, [-1, stats_embedding_size])

    target_def = states[14]
    target_def = embedding(target_def, indices=15, size=stats_embedding_size)
    target_def = tf.reshape(target_def, [-1, stats_embedding_size])

    stats = tf.concat([states[3], states[4], in_range, health_potion, bonus_potion, active_bonus_potion, agent_HP,
                       target_HP, agent_damage, target_damage, agent_def, target_def], axis=1)
    fc_stats = linear(stats, 1024, name='fc_stats', activation=tf.nn.relu)

    # Global + Local + Stats

    all_flat = tf.concat([flat_global, fc_stats, flat_local], axis=1)

    return all_flat
