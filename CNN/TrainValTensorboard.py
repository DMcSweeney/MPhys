"""Script containing tensorboard class to record losses"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.eager import context


class TrainValTensorboard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Create a log dir for Training and validation
        training_log_dir = os.path.join(log_dir, 'training')
        self.val_log_dir = os.path.join(log_dir, 'validation')
        super().__init__(training_log_dir, **kwargs)

    def set_model(self, model):
        # Setup writer for validation metrics
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
        else:
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super().set_model(model)

    def _write_custom_summaries(self, step, logs=None):
        # Define logs or type if not given
        logs = logs or {}
        # Rename key from 'val_' to ''
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)

        else:
            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        # Pass remaining logs to Tensorboard.on_epoch_end
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super()._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorboard, self).on_train_end(logs)
        self.val_writer.close()
