from layers.layers import *


# Define the input specification
def input_spec():
    position = tf.compat.v1.placeholder(tf.float32, [None, 2], name='position')
    forward_direction = tf.compat.v1.placeholder(tf.float32, [None, 1], name='forward_direction')
    target_position = tf.compat.v1.placeholder(tf.float32, [None, 2], name='target_position')
    # Qui la cell_view deve avere dim=3, quindi (None, 5, 5) e non (None, 5, 5, 1)
    # Ho fatto questo cambiamento anche nel file unity_env_wrapper.py
    cell_view = tf.compat.v1.placeholder(tf.int32, [None, 5, 5], name='cell_view')
    in_range = tf.compat.v1.placeholder(tf.float32, [None, 1], name='in_range')
    actual_potion = tf.compat.v1.placeholder(tf.float32, [None, 1], name='actual_potion')
    actual_HP = tf.compat.v1.placeholder(tf.float32, [None, 1], name='actual_HP')

    return [position, forward_direction, target_position, cell_view, in_range, actual_potion, actual_HP]


# Change the observation in real states
def obs_to_state(obs):
    position_batch = np.stack([np.asarray(state['position']) for state in obs])
    forward_direction_batch = np.stack([np.asarray(state['forward_direction']) for state in obs])
    target_position_batch = np.stack([np.asarray(state['target_position']) for state in obs])
    cell_view_batch = np.stack([np.asarray(state['cell_view']) for state in obs])
    in_range_batch = np.stack([np.asarray(state['in_range']) for state in obs])
    actual_potion_batch = np.stack([np.asarray(state['actual_potion']) for state in obs])
    actual_HP_batch = np.stack([np.asarray(state['actual_HP']) for state in obs])

    return [position_batch, forward_direction_batch, target_position_batch, cell_view_batch, in_range_batch,
            actual_potion_batch, actual_HP_batch]


# Main network specification. Usually, this network will be followed by 2 FC layers
def network_spec(states, baseline=False):

    # Aggiungo fc per aumentare la dimensione dello stato globale rispetto alla cell view
    global_state = tf.concat([states[0], states[1], states[2], states[4], states[5], states[6]], axis=1)
    fc_gs = linear(global_state, 256, name='fc_gs', activation=tf.nn.relu)

    cell_view = tf.cast(states[3], tf.int32)

    emb = embedding(cell_view, indices=4, size=32)
    conv_21 = conv_layer_2d(emb, 32, [3, 3], name='conv_21', activation=tf.nn.relu)
    conv_22 = conv_layer_2d(conv_21, 64, [3, 3], name='conv_22', activation=tf.nn.relu)
    flat_21 = tf.reshape(conv_22, [-1, 5 * 5 * 64])

    all_flat = tf.concat([fc_gs, flat_21], axis=1)

    return all_flat
