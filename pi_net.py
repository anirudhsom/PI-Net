import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras import models, layers, losses, optimizers, utils
from tensorflow.keras import backend as K


def Image_PINet():
    
    ## model
    input_shape = [32,32,3]
    initial_conv_width=3
    initial_stride=1
    initial_filters=64
    initial_pool_width=3
    initial_pool_stride=2
    use_global_pooling = True
    dropout_rate = 0.2

    model_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        128,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    x = layers.Conv2D(
        256,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    x = layers.Conv2D(
        512,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    x = layers.Conv2D(
        1024,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if use_global_pooling:
        x = layers.GlobalAveragePooling2D()(x)


    x_logits1 = layers.Dense(2500, activation="relu")(x)

    x_logits1_reshape = layers.Reshape((1,50,50))(x_logits1)

    x_logits1_reshape = layers.Permute((2,3,1))(x_logits1_reshape)

    x_logits2 = layers.Conv2DTranspose(
                            3,
                            50,
                            strides=initial_stride,
                            padding="same")(x_logits1_reshape)
    x_logits2 = layers.BatchNormalization()(x_logits2)
    x_logits2 = layers.Activation("relu")(x_logits2)

    model_output = layers.Flatten()(x_logits2)
    
    model = models.Model(model_input, model_output)

    return model

def Signal_PINet():
    
    ## model
    input_shape = [250,3]
    initial_conv_width=3
    initial_stride=1
    initial_filters=64
    initial_pool_width=3
    initial_pool_stride=2
    use_global_pooling = True
    dropout_rate = 0.2

    model_input = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        128,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    x = layers.Conv1D(
        256,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    x = layers.Conv1D(
        512,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    x = layers.Conv1D(
        1024,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if use_global_pooling:
        x = layers.GlobalAveragePooling1D()(x)


    x_logits1 = layers.Dense(2500*3, activation="relu")(x)

    x_logits1_reshape = layers.Reshape((3,50,50))(x_logits1)

    x_logits1_reshape = layers.Permute((2,3,1))(x_logits1_reshape)

    model_output = layers.Activation("relu")(x_logits1_reshape)

    model = models.Model(model_input, model_output)

    return model