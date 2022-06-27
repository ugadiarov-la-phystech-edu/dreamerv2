import tensorflow as tf
import os
from datetime import datetime


# discounting reward sequences
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


@tf.function
def make_seq_mask(mask):
    max_i = tf.argmax(mask, axis=0)
    if tf.reduce_all(mask[max_i] == 1):
        mask = tf.concat(values=(mask[:max_i], tf.ones(shape=(mask.shape[0] - max_i), dtype=mask.dtype)), axis=0)
    return tf.expand_dims(1 - mask, axis=1)


# some utilities for interpreting the trees we return
@tf.function
def build_sequences(sequences, masks, nenvs, nsteps, depth, return_mask=False, offset=0):
    # sequences are bs x size, containing e.g. rewards, actions, state reps
    # returns bs x depth x size processed sequences with a sliding window set by 'depth', padded with 0's
    # if return_mask=True also returns a mask showing where the sequences were padded
    # This can be used to produce targets for tree outputs, from the true observed sequences
    tmp_masks = tf.convert_to_tensor(masks.reshape(nenvs, nsteps).astype(int), dtype=tf.dtypes.int32)
    tmp_masks = tf.pad(tmp_masks, paddings=[[0, 0], [0, depth + offset]], mode='CONSTANT', constant_values=1)

    sequences = [tf.reshape(s, shape=(nenvs, nsteps, -1)) for s in sequences]
    if return_mask:
        mask = tf.ones_like(sequences[0], dtype=tf.dtypes.float32)
        sequences.append(mask)
    sequences = [tf.pad(s, paddings=[[0, 0], [0, depth + offset], [0, 0]], mode='CONSTANT', constant_values=1) for s in sequences]
    proc_sequences = []
    for seq in sequences:
        proc_seq = []
        for env in range(seq.shape[0]):
            for t in range(nsteps):
                seq_done_mask = make_seq_mask(tf.identity(tmp_masks[env, t+offset:t+offset+depth]))
                proc_seq.append(tf.cast(seq[env, t+offset:t+offset+depth, :], dtype=tf.dtypes.float32) * tf.cast(seq_done_mask, dtype=tf.dtypes.float32))
        proc_sequences.append(tf.stack(proc_seq))
    return proc_sequences


@tf.function
def get_paths(tree, actions, batch_size, num_actions):
    # gets the parts of the tree corresponding to actions taken
    action_indices = tf.zeros_like(actions[:, 0], dtype=tf.dtypes.int64)
    output = []
    for i, x in enumerate(tree):
        action_indices = action_indices * num_actions + actions[:, i]
        batch_indices = tf.range(batch_size, dtype=tf.dtypes.int64) * x.shape[0] // batch_size + action_indices
        output.append(tf.gather(x, batch_indices))
    return output


def get_timestamped_dir(path, name=None, link_to_latest=False):
    """Create a directory with the current timestamp."""
    current_time = datetime.now().strftime("%y-%m-%d/%H-%M-%S-%f")
    dir = path + "/" + current_time + "/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    if name is not None:
        if os.path.exists(path + "/" + name):
            os.remove(path + "/" + name)
        os.symlink(current_time, path + "/" + name, target_is_directory=True)
    if link_to_latest:
        if os.path.exists(path + "/latest"):
            os.remove(path + "/latest")
        os.symlink(current_time, path + "/latest", target_is_directory=True)
    return dir


def append_scalar(run, key, val):
    if key in run.info:
        run.info[key].append(val)
    else:
        run.info[key] = [val]


def append_list(run, key, val):
    if key in run.info:
        run.info[key].extend(val)
    else:
        run.info[key] = [val]
