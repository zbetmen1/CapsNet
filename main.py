import tensorflow as tf
import numpy as np
import utils as U
import os
from sklearn.model_selection import train_test_split
from caps_net import CapsNet
from exponential_decay import ExponentialDecay

# Meta options
U.enable_GPUs([1])
tf.keras.backend.set_image_data_format('channels_last')

# Params
n_classes = 10
input_shape = [28, 28, 1]
feature_extractor_params = {
    'filters': 256,
    'kernel_size': 9,
    'strides': 1,
    'padding': 'valid',
    'activation': 'relu'
}
primary_caps_params = {
    'capsules': 32,
    'dim': 8,
    'kernel_size': 9,
    'strides': 2,
    'padding': 'valid'
}
high_level_caps_params = {
    'capsules': n_classes,
    'dim': 16,
    'iterations': 3
}

# Loss params
mpos = 0.9
mneg = 0.1
lambd = 0.5
alpha_reconstruction = 5e-4

# Compile model
caps_net_train, caps_net_evaluate = CapsNet(input_shape, feature_extractor_params, primary_caps_params,
                                            high_level_caps_params)
caps_net_train.compile(optimizer='adam',
                       loss=[U.get_margin_loss(mpos, mneg, lambd), U.get_sse_loss(alpha_reconstruction)],
                       metrics={'magnitudes': 'accuracy'})

# Training params
batch_size = 100
epochs = 200
validation_size = 10000

lr_decay_rate = 0.96
lr_decay_step = 2000

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
model_weights_path = 'checkpoints/capsnet_decay_{}.h5'.format(lr_decay_rate)

# Callbacks
lr_decay_callback = ExponentialDecay(lr_decay_rate, lr_decay_step, verbose_steps=lr_decay_step)
checkpointer = tf.keras.callbacks.ModelCheckpoint(model_weights_path,
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  monitor='val_magnitudes_acc',
                                                  verbose=True)

# Split to train/valid
(x_train_all, y_train_all), (x_test, y_test) = U.load_mnist()
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=validation_size)

# Fit generator
hist = caps_net_train.fit_generator(U.batch_generator(x_train, y_train, batch_size),
                                    steps_per_epoch=x_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=((x_val, y_val), (y_val, x_val)),
                                    callbacks=[lr_decay_callback, checkpointer])

# Evaluate best model
caps_net_evaluate.load_weights(model_weights_path)
y_pred, _ = caps_net_evaluate.predict(x_test, batch_size=batch_size)
print('Test accuracy:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
