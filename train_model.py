import argparse
import tensorflow as tf
from meta import read_meta
from enet import get_callbacks, get_model, get_optimizer
from tfrecord import get_datasets, TRAIN_DIR, TEST_DIR


def detect_hardware():
    auto = tf.data.experimental.AUTOTUNE
    # Detect hardware, return appropriate distribution strategy
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    return strategy, auto


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=8, type=int,
                    help="Training batch size.")
    ap.add_argument("--decay", default=4, type=int,
                    help="Number of epochs between learning rate decays.")
    ap.add_argument("--dir", required=True, type=str,
                    help="Directory containing Cornell NABird Data")
    ap.add_argument("--enet", required=True, type=int,
                    help="EfficientNet Size")
    ap.add_argument("--epoch", required=True, type=int,
                    help="Number of training epochs.")
    ap.add_argument("--lr_coeff", default=1, type=float,
                    help="Coefficient of initial learning rate")
    ap.add_argument("--lr_log", default=3, type=int,
                    help="Magnitude of initial learning rate")
    ap.add_argument("--opt", default="adam", type=str,
                    help="Model optimizer")
    ap.add_argument("--res", required=True, type=int,
                    help="Image Resolution for Training")
    ap.add_argument("--weights", default="imagenet", type=str,
                    help="Pretrained Weights")

    args = vars(ap.parse_args())
    strategy, auto = detect_hardware()
    batch_size = args["batch"] * strategy.num_replicas_in_sync

    train_meta, _ = read_meta(bird_dir=args["dir"])
    category_count = train_meta["terminal_id"].nunique()

    train_files = f"{args['dir']}/{TRAIN_DIR}/*"
    test_files = f"{args['dir']}/{TEST_DIR}/*"

    train_ds, test_ds = get_datasets(train_files=train_files,
                                     test_files=test_files,
                                     category_count=category_count,
                                     c_size=args["res"],
                                     batch_size=batch_size,
                                     auto=auto)

    model = get_model(strategy, enet=args["enet"], weights=args["weights"],
                      category_count=category_count,
                      optimizer=get_optimizer(args["opt"]))
    best_model_file = make_model_file(args)
    callbacks = get_callbacks(model_file=best_model_file,
                              decay_epoch=args["decay"],
                              learning_rate=args["lr_coeff"] * 10**(-args["lr_log"]))
    model.compile(optimizer=get_optimizer(args["opt"]),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy", top_2_accuracy, top_5_accuracy])

    model.fit(train_ds,
              verbose=1,
              initial_epoch=0,
              steps_per_epoch=train_meta.shape[0] / batch_size,
              epochs=args["epoch"],
              validation_data=test_ds,
              callbacks=callbacks)


if __name__ == '__main__':
    main()