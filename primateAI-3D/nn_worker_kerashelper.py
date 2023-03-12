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


def _apply_pairwise_op(op, refTensor, compTensor):
    res = op(tf.expand_dims(tf.expand_dims(refTensor, axis=2), axis=1), tf.expand_dims(compTensor, axis=2))
    return res


def applyPairwiseOp(tensor, compRangeLength, op):
    tensorDouble = tf.concat([tensor, tensor], axis=0)

    batchSize = tf.shape(tensor)[0]

    compTensorIndices = tf.keras.backend.repeat(tf.expand_dims(tf.range(0, compRangeLength), 0), batchSize)[0]
    compTensorIndices = compTensorIndices + tf.expand_dims(tf.range(0, batchSize), 1)

    tensorDouble_comp = tf.gather(tensorDouble, compTensorIndices)
    tensor_pairwise = _apply_pairwise_op(op, tensor, tensorDouble_comp)
    tensor_pairwise_shape = tf.shape(tensor_pairwise)
    tensor_pairwise = tf.reshape(tensor_pairwise, (
    batchSize, tensor_pairwise_shape[1] * tensor_pairwise_shape[2] * tensor_pairwise_shape[3]))

    return tensor_pairwise


def gatherLoss(y_loss_ds, y_loss_pai, y_isDS, y_dsIndices, y_paiIndices):
    batchSize = tf.shape(y_isDS)[0]

    y_loss_ds_scatter = tf.scatter_nd(y_dsIndices, y_loss_ds, [batchSize])
    y_loss_pai_scatter = tf.scatter_nd(y_paiIndices, y_loss_pai, [batchSize])

    y_loss = tf.where(y_isDS, y_loss_ds_scatter, y_loss_pai_scatter)

    #y_loss = tf.Print(y_loss, [tf.reduce_sum(y_loss_ds)], message="y_loss_ds ", summarize=40)
    #y_loss = tf.Print(y_loss, [tf.reduce_sum(y_loss_pai)], message="y_loss_pai ", summarize=40)
    #y_loss = tf.Print(y_loss, [tf.reduce_sum(y_loss)], message="y_loss_all ", summarize=100)

    return y_loss


def separatePreds(y_true, y_pred):
    # batchSize = tf.shape(y_true)[0]

    y_isDS = tf.reduce_any(y_true > 5, 1)

    y_dsIndices = tf.to_int32(tf.where(y_isDS))
    y_paiIndices = tf.to_int32(tf.where(tf.logical_not(y_isDS)))

    #y_dsIndices = tf.Print(y_dsIndices, [tf.shape(y_dsIndices)[0]], message="DS samples ", summarize=10)

    y_true_ds = tf.gather_nd(y_true, y_dsIndices)
    y_pred_ds = tf.gather_nd(y_pred, y_dsIndices)

    y_true_pai = tf.gather_nd(y_true, y_paiIndices)
    y_pred_pai = tf.gather_nd(y_pred, y_paiIndices)

    return y_true_ds, y_pred_ds, y_true_pai, y_pred_pai, y_isDS, y_dsIndices, y_paiIndices


def lossLogistic_helper(x):
    x = tf.multiply(x, tf.constant(logisticCurvature))
    r = tf.add(tf.nn.relu(-x), tf.math.log1p(tf.exp(-tf.abs(x))))
    r = tf.divide(r, tf.constant(logisticCurvature))
    return r


def pairwiseLoss_helper(y_true, y_pred):
    batchSize = tf.shape(y_true)[0]

    #y_true = tf.Print(y_true, [y_true[0:3, :]], message="y_true ", summarize=40)
    #y_pred = tf.Print(y_pred, [y_pred[0:3, :]], message="y_pred ", summarize=40)

    compRangeLength_tensor = tf.constant(compRangeLength)
    compRangeLength_tensor = tf.minimum(batchSize, compRangeLength_tensor)

    y_mask = K.not_equal(y_true, mask_value)

    y_true_pairwise = applyPairwiseOp(y_true, compRangeLength_tensor, tf.subtract)
    y_pred_pairwise = applyPairwiseOp(y_pred, compRangeLength_tensor, tf.subtract)
    y_mask_pairwise = applyPairwiseOp(y_mask, compRangeLength_tensor, tf.logical_and)
    tf.stop_gradient(y_true_pairwise)

    return y_true_pairwise, y_pred_pairwise, y_mask_pairwise


def lossLogisticPairwise(y_true, y_pred):
    y_true_ds, y_pred_ds, y_true_pai, y_pred_pai, y_isDS, y_dsIndices, y_paiIndices = separatePreds(y_true, y_pred)
    y_true_pairwise, y_pred_pairwise, y_mask_pairwise = pairwiseLoss_helper(y_true_ds, y_pred_ds)
    y_loss_pai = masked_mean_squared_error(y_true_pai, y_pred_pai)

    y_true_pairwise_ceil = tf.cast(tf.greater(y_true_pairwise, tf.constant(minScoreSep)), dtype=tf.float32)
    y_true_pairwise_ceil = tf.multiply(y_true_pairwise_ceil, K.cast(y_mask_pairwise, K.floatx()))

    #y_true_pairwise_ceil = tf.Print(y_true_pairwise_ceil, [y_true_pairwise_ceil[0, :]], message="y_true_pairwise_ceil ",  summarize=40)

    y_pred_pairwise_mask = tf.multiply(y_pred_pairwise, y_true_pairwise_ceil)
    #y_pred_pairwise_mask = tf.Print(y_pred_pairwise_mask, [y_pred_pairwise_mask[0, :]], message="y_pred_pairwise_mask ", summarize=40)

    y_loss_ds = lossLogistic_helper(y_pred_pairwise_mask)

    #y_loss_ds = tf.Print(y_loss_ds, [y_loss_ds[0, :]], message="y_loss_ds_samples 1 ", summarize=40)

    y_loss_ds = tf.multiply(y_loss_ds, y_true_pairwise_ceil)
    #y_loss_ds = tf.Print(y_loss_ds, [y_loss_ds[0, :]], message="y_loss_ds_samples 2 ", summarize=40)

    y_loss_ds = tf.reduce_sum(y_loss_ds, axis=1)
    y_true_pairwise_ceil_n = tf.reduce_sum(y_true_pairwise_ceil, axis=1)
    y_true_pairwise_ceil_n = tf.maximum(tf.ones(tf.shape(y_true_pairwise_ceil_n)[0]), y_true_pairwise_ceil_n)

    y_loss_ds = tf.divide(y_loss_ds, y_true_pairwise_ceil_n)

    #y_loss_ds = tf.Print(y_loss_ds, [y_loss_ds], message="y_loss_ds ", summarize=40)

    y_loss = gatherLoss(y_loss_ds, y_loss_pai, y_isDS, y_dsIndices, y_paiIndices)

    return y_loss


def lossMSEPairwise(y_true, y_pred):
    y_true_ds, y_pred_ds, y_true_pai, y_pred_pai, y_isDS, y_dsIndices, y_paiIndices = separatePreds(y_true, y_pred)
    y_true_pairwise, y_pred_pairwise, y_mask_pairwise = pairwiseLoss_helper(y_true_ds, y_pred_ds)
    y_loss_pai = masked_mean_squared_error(y_true_pai, y_pred_pai)

    y_diff_pairwise = y_true_pairwise - y_pred_pairwise  #

    y_diff_pairwise_sq = tf.multiply(y_diff_pairwise, y_diff_pairwise)
    y_diff_pairwise_sq = tf.multiply(y_diff_pairwise_sq, K.cast(y_mask_pairwise, K.floatx()))

    y_diff_pairwise_sq = tf.where(tf.greater(y_diff_pairwise_sq, 0), y_diff_pairwise_sq, tf.zeros(tf.shape(y_diff_pairwise_sq)))
    #y_diff_pairwise_sq = tf.Print(y_diff_pairwise_sq, [y_diff_pairwise_sq[0, :]], message="y_diff_pairwise_sq ",  summarize=40)

    y_mask_pairwise_n = tf.reduce_sum(K.cast(y_mask_pairwise, K.floatx()), axis=1)
    y_mask_pairwise_n = tf.maximum(tf.ones(tf.shape(y_mask_pairwise_n)[0]), y_mask_pairwise_n)

    y_loss_ds = tf.reduce_sum(y_diff_pairwise_sq, axis=1)
    y_loss_ds = tf.divide(y_loss_ds, y_mask_pairwise_n)

    #y_loss_ds = tf.Print(y_loss_ds, [y_loss_ds], message="y_loss_ds ", summarize=40)

    y_loss = gatherLoss(y_loss_ds, y_loss_pai, y_isDS, y_dsIndices, y_paiIndices)

    return y_loss

def getLossFunDict(c):
    global compRangeLength
    global minScoreSep
    global logisticCurvature

    compRangeLength = c["compRangeLength"]
    minScoreSep = c["minScoreSep"]
    logisticCurvature = c["logisticCurvature"]

    lossFunDict = {"lossMSEPairwise": lossMSEPairwise,
                   "lossLogisticPairwise": lossLogisticPairwise,
                   "masked_loss_function": masked_loss_function,
                   "masked_binary_crossentropy": masked_binary_crossentropy,
                   "masked_mean_squared_error": masked_mean_squared_error}

    return lossFunDict


