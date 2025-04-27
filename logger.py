import tensorflow as tf
import datetime

class TensorflowLogger:
    def __init__(self, log_dir):
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.writer = tf.summary.create_file_writer(f"{log_dir}/summary_{self.timestamp}")
        self.step = 0

    def scalar(self, tag, value):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=self.step)
        self.step += 1

    def flush(self):
        self.writer.flush()