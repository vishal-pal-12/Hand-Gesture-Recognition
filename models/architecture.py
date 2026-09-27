"""
============================================================
  models/architecture.py
  Architecture implementation from the Research Paper:
  "A Deep Convolutional Neural Network Approach for Static
   Hand Gesture Recognition" (Procedia Computer Science 2020)
============================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_paper_cnn(input_shape=(100, 100, 3), num_classes=10, use_batch_norm=True):
    """
    Constructs the Deep CNN proposed by Adithya V. & Rajesh R. (2020)
    with optional Batch Normalization and He initialization for numerical
    stability when training with large spatial receptive fields (19x19, 17x17, 15x15).

    Architecture Details:
      - Input Layer: Rescaled RGB image (100 x 100 x 3)
      - Layer 1: Conv2D (8 filters, 19x19 kernel, same padding) + [BN] + ReLU + MaxPool2D ((2,2), stride 3)
      - Layer 2: Conv2D (16 filters, 17x17 kernel, same padding) + [BN] + ReLU + MaxPool2D ((2,2), stride 3)
      - Layer 3: Conv2D (32 filters, 15x15 kernel, same padding) + [BN] + ReLU + MaxPool2D ((2,2), stride 3)
      - Classification Layer: Flatten + Dense Softmax (num_classes neurons)

    Total Trainable Parameters: ~166,266 (~650 KB)
    """
    inputs = layers.Input(shape=input_shape, name="input_posture_image")

    # ----------------- FEATURE EXTRACTION LAYER 1 -----------------
    x = layers.Conv2D(
        filters=8,
        kernel_size=(19, 19),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        name='conv_layer_1'
    )(inputs)
    if use_batch_norm:
        x = layers.BatchNormalization(name='bn_layer_1')(x)
    x = layers.ReLU(name='relu_layer_1')(x)
    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(3, 3),
        padding='valid',
        name='maxpool_layer_1'
    )(x)

    # ----------------- FEATURE EXTRACTION LAYER 2 -----------------
    x = layers.Conv2D(
        filters=16,
        kernel_size=(17, 17),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        name='conv_layer_2'
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name='bn_layer_2')(x)
    x = layers.ReLU(name='relu_layer_2')(x)
    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(3, 3),
        padding='valid',
        name='maxpool_layer_2'
    )(x)

    # ----------------- FEATURE EXTRACTION LAYER 3 -----------------
    x = layers.Conv2D(
        filters=32,
        kernel_size=(15, 15),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        name='conv_layer_3'
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name='bn_layer_3')(x)
    x = layers.ReLU(name='relu_layer_3')(x)
    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(3, 3),
        padding='valid',
        name='maxpool_layer_3'
    )(x)

    # ----------------- CLASSIFICATION LAYER -----------------
    x = layers.Flatten(name='feature_flatten')(x)
    outputs = layers.Dense(
        units=num_classes,
        activation='softmax',
        name='gesture_classifier'
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='Static_Hand_Gesture_CNN_Paper_Model')
    return model

if __name__ == '__main__':
    cnn = build_paper_cnn()
    cnn.summary()
