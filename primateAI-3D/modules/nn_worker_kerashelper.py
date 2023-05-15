import sys
import json

import tensorflow as tf

from sklearn.metrics import roc_auc_score

mask_value = -1000.0
compRangeLength = None  # 50
minScoreSep = None  # 0.0
logisticCurvature = None

from keras import backend as K
import keras


def auroc(y_true, y_pred):
    y_val_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    y_val_flat = tf.where(tf.greater(y_val_flat, 5.0), y_val_flat - tf.constant(10.0), y_val_flat)

    y_val_flat_goodIdxs = tf.where(tf.not_equal(y_val_flat, -1000.0))
    y_val_flat_good = K.gather(y_val_flat, y_val_flat_goodIdxs)
    y_val_flat_good_hot = tf.where(K.greater(y_val_flat_good, 0.5), K.ones(K.shape(y_val_flat_good)),
                                   K.zeros(K.shape(y_val_flat_good)))

    y_pred_good = K.gather(y_pred_flat, y_val_flat_goodIdxs)

    hasOne = K.any(tf.equal(y_val_flat_good_hot, 1.0))
    hasZero = K.any(tf.equal(y_val_flat_good_hot, 0.0))
    hasBoth = tf.math.logical_and(hasOne, hasZero)

    y_val_flat_good_hot = K.switch(hasBoth, y_val_flat_good_hot, K.constant([1, 0]))
    y_pred_good = K.switch(hasBoth, y_pred_good, K.constant([0, 1]))

    return tf.py_func(roc_auc_score, (y_val_flat_good_hot, y_pred_good), tf.double)

def myacc(y_true, y_pred):
    y_val_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    y_val_flat_goodIdxs = tf.where(tf.not_equal(y_val_flat, -1000.0))
    y_val_flat_good = K.gather(y_val_flat, y_val_flat_goodIdxs)
    y_pred_good = K.gather(y_pred_flat, y_val_flat_goodIdxs)
    y_pred_good_hot = tf.where(K.greater(y_pred_good, 0.5), K.ones(K.shape(y_pred_good)), K.zeros(K.shape(y_pred_good)))

    nGood = K.sum(K.cast(K.equal(y_pred_good_hot, y_val_flat_good), "int32"))
    nAll = K.shape(y_val_flat_good)[0]

    acc = nGood / nAll

    return acc


def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_binary_crossentropy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_mean_squared_error(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return keras.losses.mean_squared_error(y_true * mask, y_pred * mask)


def masked_mean_squared_error(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    loss = keras.losses.mean_squared_error(y_true * mask, y_pred * mask)

    return loss


def getLossFunDict(c):
    lossFunDict = {"masked_loss_function": masked_loss_function,
                   "masked_binary_crossentropy": masked_binary_crossentropy,
                   "masked_mean_squared_error": masked_mean_squared_error}

    return lossFunDict


