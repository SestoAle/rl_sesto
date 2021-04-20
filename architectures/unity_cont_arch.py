from layers.layers import *


# Define the input specification of DeepCrawl
def input_spec():
    position = tf.compat.v1.placeholder(tf.float32, [None, 2], name='position')
    forward_direction = tf.compat.v1.placeholder(tf.float32, [None, 1], name='forward_direction')
    target_position = tf.compat.v1.placeholder(tf.float32, [None, 2], name='target_position')
    env_objects = tf.compat.v1.placeholder(tf.float32, [None, 52], name='env_objects')
    in_range = tf.compat.v1.placeholder(tf.float32, [None, 1], name='in_range')
    actual_potion = tf.compat.v1.placeholder(tf.float32, [None, 1], name='actual_potion')

    return [position, forward_direction, target_position, env_objects, in_range, actual_potion]


# Change the observation in real states
def obs_to_state(obs):
    position_batch = np.stack([np.asarray(state['position']) for state in obs])
    forward_direction_batch = np.stack([np.asarray(state['forward_direction']) for state in obs])
    target_position_batch = np.stack([np.asarray(state['target_position']) for state in obs])
    env_objects_batch = np.stack([np.asarray(state['env_objects']) for state in obs])
    in_range_batch = np.stack([np.asarray(state['in_range']) for state in obs])
    actual_potion_batch = np.stack([np.asarray(state['actual_potion']) for state in obs])

    return [position_batch, forward_direction_batch, target_position_batch, env_objects_batch, in_range_batch,
            actual_potion_batch]


# Main network specification. Usually, this network will be followed by 2 FC layers
def network_spec(states, baseline=False):

    '''
    # Cell view dovrebbe essere int16
    cell_view = states[2]

    conv_20 = embedding(cell_view, indices=3, size=32)
    conv_21 = conv_layer_2d(conv_20, 32, [3, 3], name='conv_21', activation=tf.nn.relu)
    conv_22 = conv_layer_2d(conv_21, 64, [3, 3], name='conv_22', activation=tf.nn.relu)
    flat_21 = tf.reshape(conv_22, [-1, 5 * 5 * 64])
    '''

    all_flat = tf.concat(states, axis=1)

    return all_flat
