
from globalVars import SAMPLE_JIGSAW, SAMPLE_PAI

from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import roc_auc_score
import time


class AurocEval(Callback):
    def __init__(self, c, X_val, y_val, valRows, verbose=0):

        super(AurocEval, self).__init__()
  
        self.isSampleTypeArr = np.array([tuplei[-1] for tuplei in valRows]).astype(np.int32)

        self.X_val = X_val
        self.y_val = y_val.copy()
        self.history = {}
        self.verbose = verbose


    def performEval(self, model):

        t=time.time()

        print("Starting auroc eval")

        y_pred = model.predict(self.X_val)

        metricDict = {}
        for rocType in ["plain", "pai", "jigsaw", "ds", "mean", "balanced"]:
            auc = self.getAuc(self.y_val, y_pred, rocType)
            metricDict[ "auroc_" + rocType ] = np.float64(auc)

        print("Done auroc eval (%s)" % str( time.time() - t ))
        
        return metricDict


    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        metricDict = self.performEval(self.model)

        print(metricDict)

        for sigName, sigVal in sorted(metricDict.items(), key=lambda x: x[0]):
            logs[sigName] = sigVal
            self.history.setdefault(sigName, []).append(sigVal)
            
            
    def getAuc(self, y_val_copy, y_pred, rocType):
        isPaiIdx = np.where(self.isSampleTypeArr == SAMPLE_PAI, True, False)
        isJigsawIdx = np.where(self.isSampleTypeArr == SAMPLE_JIGSAW, True, False)
        y_val = y_val_copy.copy()

        y_val = np.where((y_val >= 0.5) & (y_val < 1.0001), 1, y_val)
        y_val = np.where((y_val < 0.5) & (y_val >= 0.0), 0, y_val)

        if rocType == "plain":    
            y_val_flat = y_val.flatten()
            y_pred_flat = y_pred.flatten()

            y_val_flat_goodIdxs = np.where(np.not_equal(y_val_flat, -1000.0))[0]
            y_val_flat_good = y_val_flat[y_val_flat_goodIdxs]
            y_pred_flat_good = y_pred_flat[y_val_flat_goodIdxs]

            auc_plain = roc_auc_score( y_val_flat_good, y_pred_flat_good  )
            return auc_plain

        if rocType in ["pai", "jigsaw", "mean"]: # "ds",
            sampleTypeToRoc = {}
            for sampleTypeName, sampleTypeMask in [("pai", isPaiIdx),
                                                   ("jigsaw", isJigsawIdx)]: #("ds", isDsIdx)


                y_val_local = y_val[sampleTypeMask, :]
                y_pred_local = y_pred[sampleTypeMask, :]

                y_val_flat = y_val_local.flatten()
                y_pred_flat = y_pred_local.flatten()

                y_val_flat_goodIdxs = np.where(np.not_equal(y_val_flat, -1000.0))[0]
                y_val_flat_good = y_val_flat[y_val_flat_goodIdxs]
                y_pred_flat_good = y_pred_flat[y_val_flat_goodIdxs]

                auc = 0
                if y_val_flat_good.shape[0] > 10:
                    auc = roc_auc_score(y_val_flat_good, y_pred_flat_good)

                sampleTypeToRoc[sampleTypeName] = auc

            if rocType == "mean":
                aucs = []
                for sampleTypeName, auc in sampleTypeToRoc.items():
                    if auc == 0:
                        return 0.0
                    else:
                        aucs.append(auc)

                auc_mean = sum(aucs) / len(aucs)
                return auc_mean
            else:
                return sampleTypeToRoc[rocType]

        if rocType == "balanced":

            sampleSizes = []
            for sampleTypeName, sampleTypeMask in [("pai", isPaiIdx),
                                                   ("jigsaw", isJigsawIdx)]:

                y_val_local = y_val[sampleTypeMask, :]
                y_val_flat = y_val_local.flatten()
                y_val_flat_goodIdxs = np.where(np.not_equal(y_val_flat, -1000.0))[0]

                if y_val_flat_goodIdxs.shape[0] == 0:
                    return 0.0

                sampleSizes.append(y_val_flat_goodIdxs.shape[0])

            targetSampleSize = min(sampleSizes)

            y_val_flat_good_list = []
            y_pred_flat_good_list = []

            for sampleTypeName, sampleTypeMask in [("pai", isPaiIdx),
                                                   ("jigsaw", isJigsawIdx)]:

                y_val_local = y_val[sampleTypeMask, :]
                y_val_flat = y_val_local.flatten()
                y_val_flat_goodIdxs = np.where(np.not_equal(y_val_flat, -1000.0))[0]
                y_val_flat_goodIdxs_sample = np.random.choice(y_val_flat_goodIdxs,
                                                                     size=targetSampleSize,
                                                                     replace=False)

                y_pred_local = y_pred[sampleTypeMask, :]
                y_pred_flat = y_pred_local.flatten()

                y_val_flat_good = y_val_flat[y_val_flat_goodIdxs_sample]
                y_pred_flat_good = y_pred_flat[y_val_flat_goodIdxs_sample]

                y_val_flat_good_list.append( y_val_flat_good )
                y_pred_flat_good_list.append( y_pred_flat_good )

            y_val_flat_good = np.concatenate(y_val_flat_good_list)
            y_pred_flat_good = np.concatenate(y_pred_flat_good_list)

            auc_balanced = roc_auc_score(y_val_flat_good, y_pred_flat_good)

            return auc_balanced


