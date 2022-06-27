import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import keras.initializers as initializers


def build_transition_fn(embedding_dim, num_actions=3):
    transition_fun1 = layers.Dense(embedding_dim, activation='linear')
    transition_fun2 = tf.Variable(
        tf.cast(initializers.GlorotNormal()(shape=(embedding_dim, embedding_dim, num_actions)), dtype=tf.dtypes.float32),
        dtype=tf.dtypes.float32
    )
    return transition_fun1, transition_fun2


class MLPRewardFn(keras.Model):
    def __init__(self, embed_dim, num_actions):
        super(MLPRewardFn, self).__init__()
        self.embedding_dim = embed_dim
        self.num_actions = num_actions

        self.mlp = keras.Sequential([
            layers.Dense(64, activation='relu', kernel_initializer=initializers.Orthogonal(gain=2 ** 0.5)),
            layers.Dense(num_actions, activation='linear', kernel_initializer=initializers.Orthogonal(gain=0.01))
        ])

    @tf.function
    def call(self, x):
        x = tf.reshape(x, [-1, self.embedding_dim])
        return tf.reshape(self.mlp(x), [-1, self.num_actions])
