import tensorflow as tf
import numpy as np
from primary_caps import PrimaryCapsConv2D
from high_level_caps import HighLevelCaps


def CapsNet(input_shape, feature_extractor_params, primary_caps_params, high_level_caps_params):
    # Inputs to models
    n_classes = high_level_caps_params['capsules']
    img = tf.keras.layers.Input(shape=input_shape)
    labels = tf.keras.layers.Input(shape=(n_classes,))

    # Build base for models
    features = tf.keras.layers.Conv2D(filters=feature_extractor_params['filters'],
                                      kernel_size=feature_extractor_params['kernel_size'],
                                      strides=feature_extractor_params['strides'],
                                      padding=feature_extractor_params['padding'],
                                      activation=feature_extractor_params['activation'],
                                      name='conv1')(img)

    primary_caps_out = PrimaryCapsConv2D(primary_caps_params['capsules'],
                                         primary_caps_params['dim'],
                                         primary_caps_params['kernel_size'],
                                         primary_caps_params['strides'],
                                         primary_caps_params['padding'],
                                         name='primary_caps')(features)

    high_caps_out = HighLevelCaps(high_level_caps_params['capsules'], high_level_caps_params['dim'],
                                  high_level_caps_params['iterations'], name='high_level_caps')(primary_caps_out)

    magnitudes = tf.keras.layers.Lambda(lambda vectors: tf.sqrt(tf.reduce_sum(high_caps_out * high_caps_out, axis=-1)),
                                        name='magnitudes')(
        high_caps_out)

    # Build reconstruction decoder for regularization
    reconstruction = tf.keras.models.Sequential(name='reconstruction')
    reconstruction.add(tf.keras.layers.Dense(512, activation='relu', input_dim=high_level_caps_params['capsules'] *
                                                                        high_level_caps_params['dim']))
    reconstruction.add(tf.keras.layers.Dense(1024, activation='relu'))
    reconstruction.add(tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid'))
    reconstruction.add(tf.keras.layers.Reshape(target_shape=input_shape))

    # Create two masked outputs from base model, first using labels for training and second using index of capsule
    # with highest activation (magnitude)
    def mask_with_labels(args):
        caps_out, targets = args
        masked = caps_out * targets[:, :, None]
        return tf.reshape(masked, [-1, tf.reduce_prod(tf.shape(masked)[1:])])

    masked_with_labels = tf.keras.layers.Lambda(mask_with_labels)([high_caps_out, labels])

    def mask_with_magnitudes(args):
        caps_out, lengths = args
        mask = tf.one_hot(tf.argmax(lengths, axis=-1), depth=n_classes, axis=-1)
        masked = caps_out * tf.cast(mask[:, :, None], tf.float32)
        return tf.reshape(masked, [-1, tf.reduce_prod(tf.shape(masked)[1:])])

    masked_with_magnitudes = tf.keras.layers.Lambda(mask_with_magnitudes)([high_caps_out, magnitudes])

    # Build two models, first for training and second for prediction, share weights
    caps_net_train = tf.keras.models.Model([img, labels], [magnitudes, reconstruction(masked_with_labels)])
    caps_net_evaluate = tf.keras.models.Model(img, [magnitudes, reconstruction(masked_with_magnitudes)])
    return caps_net_train, caps_net_evaluate
