import tensorflow as tf
import os

def enable_GPUs(gpu_list):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for gpu in gpu_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def squash(s):
    s_norm2 = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
    scale = (s_norm2 / (1 + s_norm2)) / tf.sqrt(s_norm2 + 1e-7)
    return scale * s


def get_margin_loss(mpos, mneg, lambd):
    def margin_loss(y_true, y_pred):
        L = y_true * tf.square(tf.maximum(0., mpos - y_pred)) + \
            lambd * (1 - y_true) * tf.square(tf.maximum(0., y_pred - mneg))

        return tf.reduce_sum(L)

    return margin_loss


def get_sse_loss(alpha):
    def sse_loss(y_true, y_pred):
        L = tf.reduce_sum(tf.square(y_true - y_pred))
        return L * alpha

    return sse_loss


def batch_generator(x_train, y_train, batch_size, pixel_shift=2):
    generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=pixel_shift + 1,
                                                                height_shift_range=pixel_shift + 1)
    iterator = generator.flow(x_train, y_train, batch_size=batch_size)
    while True:
        x_batch, y_batch = iterator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])  # output must be tuple (inputs, targets);


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = tf.keras.utils.to_categorical(y_train.astype('float32'))
    y_test = tf.keras.utils.to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)
