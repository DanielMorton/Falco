import tensorflow as tf
from .efficientnet import efficientnet
from enet import DROPOUT


def make_model_file(args):
    """Constructs file name for saving model.

    :param args: Dictionary of command line arguments.
    :type args: dict

    :return: Name for saved model file.
    :rtype: str
    """
    best_model_file = f"{args['dir']}/"

    # Add EfficientNet Size.
    best_model_file += f"enet{args['enet']}"

    # Add Image Resolution.
    best_model_file += f"_r{args['res']}"

    # Add initial learning rate magnitude.
    best_model_file += f"_{args['lr_log']}"

    # Add initial learning rate coefficient.
    best_model_file += f"_{args['lr_coeff']}"

    # Add learning rate decay.
    best_model_file += f"_{args['decay']}.h5"
    return best_model_file


def top_2_accuracy(y_true, y_pred):
    """Wrapper function for computing top 2 accuracy."""
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 2)


def top_5_accuracy(y_true, y_pred):
    """Wrapper function for computing top 5 accuracy."""
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 5)


def get_model(strategy,
              enet,
              weights,
              category_count,
              optimizer):
    """Constructs the full model from EfficientNet base and new softmax.

    :param strategy: Tensorflow distribution strategy. CPU vs GPU vs TPU
    :param enet: Size of EfficientNet backbone.
    :type enet: int
    :param weights: Pretrained weight for EfficientNet.
    :type weights: str
    :param category_count: Number of categories to predict.
    :type category_count: int
    :param optimizer: Optimizer used for training.
    :type optimizer: tf.keras.optimizers.Optimizer

    :return Model for training.
    :rtype: tf.keras.models.Model
    """

    with strategy.scope():
        base_model = efficientnet(b=enet,
                                  weights=weights,
                                  include_top=False)

        # Add GAP layer.
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

        # Add dropout appropriate to model size.
        output = tf.keras.layers.Dropout(DROPOUT[enet])(output)

        # Add softmax output layer.
        output = tf.keras.layers.Dense(category_count,
                                       activation='softmax',
                                       name='name')(output)
        enet = tf.keras.models.Model(base_model.input,
                                     outputs=output)
        enet.compile(optimizer=optimizer,
                     loss=tf.keras.losses.CategoricalCrossentropy(),
                     metrics=["accuracy", top_2_accuracy, top_5_accuracy])
        return enet
