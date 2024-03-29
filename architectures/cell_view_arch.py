from layers.layers import *


# Define the input specification
def input_spec():
    position = tf.compat.v1.placeholder(tf.float32, [None, 2], name='position')
    forward_direction = tf.compat.v1.placeholder(tf.float32, [None, 1], name='forward_direction')
    target_position = tf.compat.v1.placeholder(tf.float32, [None, 2], name='target_position')
    differences = tf.compat.v1.placeholder(tf.float32, [None, 2], name='differences')
    # Qui la cell_view deve avere dim=3, quindi (None, 5, 5) e non (None, 5, 5, 1)
    # Ho fatto questo cambiamento anche nel file unity_env_wrapper.py
    cell_view = tf.compat.v1.placeholder(tf.int32, [None, 7, 7], name='cell_view')
    in_range = tf.compat.v1.placeholder(tf.float32, [None, 1], name='in_range')
    actual_potion = tf.compat.v1.placeholder(tf.float32, [None, 1], name='actual_potion')
    agent_actual_HP = tf.compat.v1.placeholder(tf.float32, [None, 1], name='agent_actual_HP')
    target_actual_HP = tf.compat.v1.placeholder(tf.float32, [None, 1], name='target_actual_HP')

    return [position, forward_direction, target_position, differences, cell_view, in_range, actual_potion,
            agent_actual_HP, target_actual_HP]


# Change the observation in real states
def obs_to_state(obs):
    position_batch = np.stack([np.asarray(state['position']) for state in obs])
    forward_direction_batch = np.stack([np.asarray(state['forward_direction']) for state in obs])
    target_position_batch = np.stack([np.asarray(state['target_position']) for state in obs])
    differences_batch = np.stack([np.asarray(state['differences']) for state in obs])
    cell_view_batch = np.stack([np.asarray(state['cell_view']) for state in obs])
    in_range_batch = np.stack([np.asarray(state['in_range']) for state in obs])
    actual_potion_batch = np.stack([np.asarray(state['actual_potion']) for state in obs])
    agent_actual_HP_batch = np.stack([np.asarray(state['agent_actual_HP']) for state in obs])
    target_actual_HP_batch = np.stack([np.asarray(state['target_actual_HP']) for state in obs])

    return [position_batch, forward_direction_batch, target_position_batch, differences_batch, cell_view_batch,
            in_range_batch, actual_potion_batch, agent_actual_HP_batch, target_actual_HP_batch]


# Main network specification. Usually, this network will be followed by 2 FC layers
def network_spec(states, baseline=False):

    # Aggiungo fc per aumentare la dimensione dello stato globale rispetto alla cell view
    global_state = tf.concat([states[0], states[1], states[2], states[3], states[6], states[7], states[8]], axis=1)
    fc_gs = linear(global_state, 256, name='fc_gs', activation=tf.nn.relu)

    cell_view = tf.cast(states[4], tf.int32)

    emb = embedding(cell_view, indices=4, size=16)
    conv_21 = conv_layer_2d(emb, 16, [3, 3], name='conv_21', activation=tf.nn.relu)
    conv_22 = conv_layer_2d(conv_21, 32, [3, 3], name='conv_22', activation=tf.nn.relu)
    flat_21 = tf.reshape(conv_22, [-1, 7 * 7 * 32])

    all_flat = tf.concat([fc_gs, flat_21], axis=1)

    return all_flat