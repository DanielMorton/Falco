import tensorflow as tf
import tensorflow_addons as tfa
from enet import EPSILON, MOMENTUM


def get_optimizer(opt):
    if opt == 'rms':
        return tf.keras.optimizers.RMSprop(momentum=MOMENTUM,
                                           epsilon=EPSILON)
    elif opt == 'adam':
        return tf.keras.optimizers.Adam()
    elif opt == 'radam':
        return tfa.optimizers.RectifiedAdam()
    elif opt == 'adamw':
        return tfa.optimizers.AdamW(weight_decay=1e-5)
