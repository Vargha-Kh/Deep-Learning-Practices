from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


Callbacks = [LRTensorBoard(log_dir='./DIRECTORY')]
