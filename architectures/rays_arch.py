from layers.layers import *


# Define the input specification
def input_spec():
    position = tf.compat.v1.placeholder(tf.float32, [None, 2], name='position')
    forward_direction = tf.compat.v1.placeholder(tf.float32, [None, 1], name='forward_direction')
    target_position = tf.compat.v1.placeholder(tf.float32, [None, 2], name='target_position')
    rays = tf.compat.v1.placeholder(tf.float32, [None, 36, 5], name='rays')
    in_range = tf.compat.v1.placeholder(tf.float32, [None, 1], name='in_range')
    actual_potion = tf.compat.v1.placeholder(tf.float32, [None, 1], name='actual_potion')

    return [position, forward_direction, target_position, rays, in_range, actual_potion]


# Change the observation in real states
def obs_to_state(obs):
    position_batch = np.stack([np.asarray(state['position']) for state in obs])
    forward_direction_batch = np.stack([np.asarray(state['forward_direction']) for state in obs])
    target_position_batch = np.stack([np.asarray(state['target_position']) for state in obs])
    rays_batch = np.stack([np.asarray(state['rays']) for state in obs])
    in_range_batch = np.stack([np.asarray(state['in_range']) for state in obs])
    actual_potion_batch = np.stack([np.asarray(state['actual_potion']) for state in obs])

    return [position_batch, forward_direction_batch, target_position_batch, rays_batch, in_range_batch,
            actual_potion_batch]


# Main network specification. Usually, this network will be followed by 2 FC layers
def network_spec(states, baseline=False):

    # Aggiungo fc per aumentare la dimensione dello stato globale rispetto alla cell view
    global_state = tf.concat([states[0], states[1], states[2], states[4], states[5]], axis=1)
    fc_gs = linear(global_state, 256, name='fc_gs', activation=tf.nn.relu)

    conv1d = circ_conv1d(states[3], filters=16, kernel_size=3, name='conv1d', activation='relu')
    flat_conv1d = tf.reshape(conv1d, [-1, 16 * 36])

    all_flat = tf.concat([fc_gs, flat_conv1d], axis=1)

    return all_flat
