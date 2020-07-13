import tensorflow as tf


def get_callbacks(model_file,
                  decay_epoch,
                  learning_rate):
    """Returns the callbacks used for training Cornell NA Birds.
    Three callbacks are used:
    CSVLogger - writes logs after each epoch
    ModelCheckpoint - Saves model if validation log_loss improves.
    LearningRateScheduler - Adjusts the learning rate after each epoch.

    :param model_file: File name of the saved model.
    :type model_file: str
    :param decay_epoch: Number of epochs between learning rate decay.
    :type decay_epoch: int
    :param learning_rate: Initial learning rate.
    :type learning_rate: float

    :return: List of callbacks.
    :rtype: List[tf.keras.callbacks.Callback]
    """

    def scheduler(epoch):
        return 0.94 ** (epoch // decay_epoch) * learning_rate

    return [tf.keras.callbacks.CSVLogger(model_file.replace('.h5', '.csv')),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_file,
                                               verbose=1,
                                               save_best_only=True,
                                               mode="auto"),
            tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)]
