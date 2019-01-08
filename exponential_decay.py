import tensorflow as tf

class ExponentialDecay(tf.keras.callbacks.Callback):
    def __init__(self, decay_rate, decay_steps, staircase=False, verbose_steps=None):
        super(ExponentialDecay, self).__init__()
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.staircase = staircase

        self.step = 0
        self.base_lr = None

        self.verbose_steps = verbose_steps

    def on_epoch_begin(self, epoch, logs=None):
        # Check for learning rate
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Grab base learning rate
        if self.base_lr is None:
            self.base_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        # Print base learning rate if verbose
        if self.verbose_steps is not None and epoch == 0:
            print('Exponential decay: base learning rate {}.'.format(self.base_lr))

    def on_batch_begin(self, batch, logs=None):
        # Check for learning rate
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Apply decay formula and set model learning rate
        power = self.step / self.decay_steps if not self.staircase else self.step // self.decay_steps
        lr = self.base_lr * (self.decay_rate ** power)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        # Update step and print if verbose
        self.step += 1
        if self.verbose_steps is not None and (self.step % self.verbose_steps) == 0:
            print('\nExponential decay: current step {}; current learning rate {}.'.format(self.step, lr))

