import tensorflow as tf
import numpy as np


def MLP(name, inputs, nums_in, nums_out):
    inputs = tf.layers.flatten(inputs)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.random_normal_initializer(stddev=0.08))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer(0.))
        inputs = tf.matmul(inputs, W) + b
    return inputs

def preprocess(data):
    return (data / 255.0 - 0.5) * 2