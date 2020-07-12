import tensorflow as tf


def get_callbacks(model_file,
                  decay_epoch,
                  learning_rate):
    def scheduler(epoch):
        return 0.94 ** (epoch // decay_epoch) * learning_rate

    return [tf.keras.callbacks.CSVLogger(model_file.replace('.h5', '.csv')),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_file,
                                               verbose=1,
                                               save_best_only=True,
                                               mode="auto"),
            tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)]
