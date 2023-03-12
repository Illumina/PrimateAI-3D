from nn_worker_helper import getVoxelsForSnpRows_tripleBased_orderSafe
from keras.callbacks import Callback
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import os

import time

class CorrelationEval(Callback):
    def __init__(self, c, variantsDfFilePath, pdbLmdb, dsDF, verbose=0):

        super(CorrelationEval, self).__init__()
        
        print(variantsDfFilePath)
        snpDF = pd.read_csv(variantsDfFilePath)

        if not dsDF is None:
            print("Reducing to DS variants")
            print("Before: ", len(snpDF))
            snpDF = snpDF.merge(dsDF, on=dsDF.columns.tolist())
            print("After: ", len(snpDF))

        snpArray_tmp = snpDF[["gene_name", "change_position_1based", "label_numeric_aa", "label_numeric_aa_alt"]].values

        snpRows = []
        snpRows_jigsaw = []
        for name, change_position_1based, label_numeric_aa, label_numeric_aa_alt in snpArray_tmp:
            labelArr = np.zeros(20, dtype=np.float32)

            isJigsaw = False
            snpRows.append( (name, change_position_1based, label_numeric_aa, labelArr, isJigsaw) )
            isJigsaw = True
            snpRows_jigsaw.append( (name, change_position_1based, label_numeric_aa, labelArr, isJigsaw) )

            

        X_eval, y_eval, countTuples, _ = getVoxelsForSnpRows_tripleBased_orderSafe(snpRows, c, pdbLmdb)
        X_eval_jigsaw, y_eval_jigsaw, countTuples, _ = getVoxelsForSnpRows_tripleBased_orderSafe(snpRows_jigsaw, c, pdbLmdb)

        self.y_eval = y_eval

        
        self.evalName = os.path.basename(variantsDfFilePath).replace(".csv", "")
        
        self.X_eval = X_eval
        self.X_eval_jigsaw = X_eval_jigsaw
        self.y_score_assay = snpDF.assay_score.values
        
        
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.snpDF = snpDF
        

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}


    def performEval(self, model, savePredFile=None):
        t=time.time()

        print("Starting corr eval")

        y_pred = model.predict(self.X_eval)
        y_pred_jigsaw = model.predict(self.X_eval_jigsaw)
        y_pred_both = np.mean([y_pred, y_pred_jigsaw], axis=0)

        alt_score = y_pred[ np.arange(y_pred.shape[0]), self.snpDF.label_numeric_aa_alt.values ]
        alt_score_jigsaw = y_pred_jigsaw[ np.arange(y_pred_jigsaw.shape[0]), self.snpDF.label_numeric_aa_alt.values ]
        alt_score_both = y_pred_both[ np.arange(y_pred_both.shape[0]), self.snpDF.label_numeric_aa_alt.values ]

        y_df = pd.DataFrame({"id": self.snpDF.snp_id, "act": self.y_score_assay, "pred": alt_score, "predJigsaw": alt_score_jigsaw, "predBoth": alt_score_both})

        corr, sig = spearmanr(y_df.act, y_df.pred)
        corrJigsaw, sigJigsaw = spearmanr(y_df.act, y_df.predJigsaw)
        corrBoth, sigBoth = spearmanr(y_df.act, y_df.predBoth)
        
        sigDict = { "sig_score_raw_alt_%s" % self.evalName: np.log10(sig),
                    "corr_score_raw_alt_%s" % self.evalName: corr,
                    "sig_score_rawJigsaw_alt_%s" % self.evalName: np.log10(sigJigsaw),
                    "corr_score_rawJigsaw_alt_%s" % self.evalName: corrJigsaw,
                    "sig_score_rawBoth_alt_%s" % self.evalName: np.log10(sigBoth),
                    "corr_score_rawBoth_alt_%s" % self.evalName: corrBoth  }

        if savePredFile != None:
            y_df.to_csv(savePredFile)

        print("Done corr eval (%s)" % str( time.time() - t ))
        
        return y_df, sigDict 


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        y_df, sigDict = self.performEval(self.model)

        for sigName, sigVal in sorted(sigDict.items(), key=lambda x: x[0]):
            logs[sigName] = sigVal
            self.history.setdefault(sigName, []).append(sigVal)


