import argparse
import numpy as np
import tensorflow as tf
from enet import detect_hardware, make_model_file, top_2_accuracy, top_5_accuracy
from meta import read_meta
from tfrecord import get_test_datasets, TEST_DIR


def main():
    """Evaluates selected model on NABirds Test data. Outputs accuracy to console."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default=False, type=bool,
                    help="Use bounding box crops for evaluation.")
    ap.add_argument("--decay", default=4, type=int,
                    help="Number of epochs between learning rate decays.")
    ap.add_argument("--dir", required=True, type=str,
                    help="Directory containing Cornell NABird Data")
    ap.add_argument("--enet", required=True, type=int,
                    help="EfficientNet Size")
    ap.add_argument("--free", default=False, type=bool,
                    help="Don't rescale images to fixed size.")
    ap.add_argument("--lr_coeff", default=1, type=float,
                    help="Coefficient of initial learning rate")
    ap.add_argument("--lr_log", default=3, type=int,
                    help="Magnitude of initial learning rate")
    ap.add_argument("--res", required=True, type=int,
                    help="Image Resolution for Testing")

    args = vars(ap.parse_args())
    train_meta, _ = read_meta(bird_dir=args["dir"])
    category_count = train_meta["terminal_id"].nunique()
    strategy, auto = detect_hardware()
    batch_size = args["batch"] * strategy.num_replicas_in_sync
    model_file = make_model_file(args)
    model = tf.keras.models.load_model(model_file,
                                       custom_objects={"top_2_accuracy": top_2_accuracy,
                                                       "top_5_accuracy": top_5_accuracy})
    test_files = f"{args['dir']}/{TEST_DIR}/*"
    test_data = get_test_datasets(test_files, batch_size=batch_size,
                                  category_count=category_count,
                                  resolution=args["res"],
                                  auto=auto)

    ground_truth = np.concatenate([v.numpy() for v in list(test_data.map(lambda image, label: label))])
    pred = model.predict(test_data.map(lambda image, label: image))
    correct_pred = tf.keras.metrics.categorical_accuracy(ground_truth, pred)
    accuracy = tf.math.reduce_sum(correct_pred).numpy()/ground_truth.shape[0]
    print(f"Model accuracy = {100* np.round(accuracy, 3)}")


if __name__ == '__main__':
    main()
