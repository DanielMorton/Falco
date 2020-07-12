import tensorflow as tf
from .efficientnet import efficientnet
from enet import DROPOUT


def make_model_file(args):
    best_model_file = f"{args['dir']}/"
    best_model_file += f"enet{args['enet']}"
    best_model_file += f"_r{args['res']}"
    best_model_file += f"_{args['lr_log']}"
    best_model_file += f"_{args['lr_coeff']}"
    best_model_file += f"_{args['decay']}.h5"
    return best_model_file


def top_2_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 2)


def top_5_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 5)


def get_model(strategy, enet, weights, category_count, optimizer):
    with strategy.scope():
        base_model = efficientnet(b=enet,
                                  weights=weights,
                                  include_top=False)
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dropout(DROPOUT[enet])(output)
        output = tf.keras.layers.Dense(category_count,
                                       activation='softmax',
                                       name='name')(output)
        enet = tf.keras.models.Model(base_model.input, outputs=output)
        enet.compile(optimizer=optimizer,
                     loss=tf.keras.losses.CategoricalCrossentropy(),
                     metrics=["accuracy", top_2_accuracy, top_5_accuracy])
        return enet
