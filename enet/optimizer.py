import tensorflow as tf
import tensorflow_addons as tfa
from enet import EPSILON, MOMENTUM


def get_optimizer(opt):
    """Returns the optimizer to use in training.

    :param opt: Optimizer name.
    :type opt: str

    :return: Optimizer for training.
    :rtype: tf.keras.optimizers.Optimizer
    """

    if opt == 'rms':
        return tf.keras.optimizers.RMSprop(momentum=MOMENTUM,
                                           epsilon=EPSILON)
    elif opt == 'adam':
        return tf.keras.optimizers.Adam()
    elif opt == 'radam':
        return tfa.optimizers.RectifiedAdam()
    elif opt == 'adamw':
        return tfa.optimizers.AdamW(weight_decay=1e-5)
