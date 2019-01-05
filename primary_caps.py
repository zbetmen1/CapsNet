import tensorflow as tf
import utils as U


class PrimaryCapsConv2D(tf.keras.layers.Conv2D):
    def __init__(self, n_caps, caps_dim, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsConv2D, self).__init__(n_caps * caps_dim, kernel_size, strides=strides, padding=padding,
                                                **kwargs)

        self.n_caps = n_caps
        self.caps_dim = caps_dim

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        conv2d_outputs = super(PrimaryCapsConv2D, self).call(inputs)
        conv2d_outputs_int_shape = conv2d_outputs.get_shape().as_list()
        caps_outputs = conv2d_outputs_int_shape[1] * conv2d_outputs_int_shape[2] * self.n_caps
        s = tf.reshape(conv2d_outputs, [batch_size, caps_outputs, self.caps_dim])
        v = U.squash(s)
        return v

    def compute_output_shape(self, input_shape):
        conv2d_output_shape = super(PrimaryCapsConv2D, self).compute_output_shape(input_shape)
        return [conv2d_output_shape[0], conv2d_output_shape[1] * conv2d_output_shape[2] * self.n_caps, self.caps_dim]

    def get_config(self):
        config = {
            'n_caps': self.n_caps,
            'caps_dim': self.caps_dim,
        }
        base_config = super(PrimaryCapsConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
