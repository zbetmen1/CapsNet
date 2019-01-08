import tensorflow as tf
import utils as U

class HighLevelCaps(tf.keras.layers.Layer):
    def __init__(self, n_caps, caps_dim, routing_iters, kernel_initializer='glorot_uniform', **kwargs):
        super(HighLevelCaps, self).__init__(**kwargs)

        self.n_caps = n_caps
        self.caps_dim = caps_dim
        self.routing_iters = routing_iters
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get('zeros')

    def build(self, input_shape):
        assert len(input_shape) == 3, 'Input must have shape (B, incaps, indim)'

        prev_caps_dim = int(input_shape[-1])
        prev_caps_outputs = int(input_shape[1])
        self.W = self.add_weight(name='W', shape=[prev_caps_outputs, self.n_caps, self.caps_dim, prev_caps_dim],
                                 initializer=self.kernel_initializer)
        self.bias = self.add_weight(name='bias', shape=[self.n_caps, self.caps_dim], initializer=self.bias_initializer)

        # Mandatory call specified in Keras docs
        super(HighLevelCaps, self).build(input_shape)

    def call(self, u, **kwargs):
        # Wij @ ui for every batch and every i, j pair; Shapes are as follows:
        # shape(W) = [incaps(i), outcaps(o), outdim(m), indim(n)]
        # shape(u) = [B(b), incaps(i), indim(n)]
        # shape(uhat) = [B(b), incaps(i), outcaps(o), outdim(m)]
        uhat = tf.einsum('iomn,bin->biom', self.W, u)

        # Prepare logits for routing; shape of logits is [B, incaps, outcaps]
        u_shape = u.get_shape().as_list()
        batch_size = tf.shape(u)[0]
        prev_caps = u_shape[1]
        b = tf.zeros([batch_size, prev_caps, self.n_caps], tf.float32)
        for i in range(self.routing_iters):
            # We compute softmax over outcaps (i.e. c(i,j) = exp(b(i, j)) / sum_k{exp(b(i, k))})
            c = tf.nn.softmax(b, axis=-1)

            # Next, compute s(j) = sum_i{c(i,j)*uhat(j|i)}; shape of s is [b, outcaps, outdim]
            # NOTE: Adding self.bias is extension not documented in a paper (probably as implementation detail)
            s = tf.einsum('biom,bio->bom', uhat, c) + self.bias

            # Squash s to get v; same shape as s
            v = U.squash(s)

            # If not last iteration add contribution to logits; b(i,j) += v(j) . uhat(j|i)
            if i != self.routing_iters-1:
                b += tf.einsum('biom,bom->bio', uhat, v)

        return v

    def compute_output_shape(self, input_shape):
        return [None, self.n_caps, self.caps_dim]

    def get_config(self):
        config = {
            'n_caps': self.n_caps,
            'caps_dim': self.caps_dim,
            'routing_iters': self.routing_iters
        }
        base_config = super(HighLevelCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
