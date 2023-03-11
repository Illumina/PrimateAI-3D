from nn_worker_helper import getVoxelsForSnpRows_tripleBased_orderSafe
from keras.callbacks import Callback
from scipy.stats import ranksums
import pandas as pd
import numpy as np
import os
from keras.utils import Sequence, np_utils
import time


class RanksumEval(Callback):
    def __init__(self, c, variantsDfFilePath, pdbLmdb, dsDF, verbose=0):
        super(RanksumEval, self).__init__()

        dddDF = pd.read_csv(variantsDfFilePath, index_col=0)

        if not dsDF is None:
            print("Reducing to DS variants")
            print("Before: ", len(dddDF))
            dddDF = dddDF.merge(dsDF, on=dsDF.columns.tolist())
            print("After: ", len(dddDF))

        dddDF = dddDF[~dddDF["name"].isna()].copy()

        dddRows_tmp = dddDF[["name", "change_position_1based", "label_numeric_aa", "label_numeric_aa_alt"]].values.tolist() #

        dddRows = []
        dddRows_jigsaw = []
        for name, change_position_1based, label_numeric_aa, label_numeric_aa_alt in dddRows_tmp:
            labelArr = np.zeros(20, dtype=np.float32)
            labelArr[ label_numeric_aa ] = 1.0
            labelArr[ label_numeric_aa_alt ] = 1.0

            isJigsaw = False
            dddRows.append( (name, change_position_1based, label_numeric_aa, labelArr, isJigsaw) )
            isJigsaw = True
            dddRows_jigsaw.append( (name, change_position_1based, label_numeric_aa, labelArr, isJigsaw) )


        X_eval, y_eval, countTuples, _ = getVoxelsForSnpRows_tripleBased_orderSafe(dddRows, c, pdbLmdb)
        X_eval_jigsaw, y_eval_jigsaw, countTuples, _ = getVoxelsForSnpRows_tripleBased_orderSafe(dddRows_jigsaw, c, pdbLmdb)

        self.y_eval = y_eval
        self.y_ddd = np_utils.to_categorical(dddDF.label_numeric_func, num_classes=2)

        self.evalName = os.path.basename(variantsDfFilePath).replace(".csv", "")
        
        self.X_eval = X_eval
        self.X_eval_jigsaw = X_eval_jigsaw
        
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.snpDF = dddDF


    def performEvalBinary(self, model, savePredFile=None):
        t=time.time()

        print("Starting ranksum eval")
        y_pred_eval = model.predict(self.X_eval)
        

        y_df = pd.DataFrame({"id": self.snpDF.snp_id, "act": self.y_ddd[:, 0], "pred": y_pred_eval[:, 0]})

        sig = ranksums(y_df[ y_df.act == 1.0 ].pred, 
                       y_df[ y_df.act == 0.0 ].pred).pvalue

        sigDict = {"score_raw_alt": np.log10(sig)}

        if savePredFile != None:
            y_df.to_csv(savePredFile)
        
        print("Done ranksum eval (%s)" % str( time.time() - t ))

        return y_df, sigDict

    def performEval(self, model, savePredFile=None):

        t=time.time()

        print("Starting ranksum eval")

        y_pred_eval = model.predict(self.X_eval)
        y_pred_eval_jigsaw = model.predict(self.X_eval_jigsaw)
        y_pred_eval_both = np.mean([y_pred_eval, y_pred_eval_jigsaw], axis=0)

        alt_score = y_pred_eval[ np.arange(y_pred_eval.shape[0]), self.snpDF.label_numeric_aa_alt.values ]
        alt_score_jigsaw = y_pred_eval_jigsaw[ np.arange(y_pred_eval_jigsaw.shape[0]), self.snpDF.label_numeric_aa_alt.values ]
        alt_score_both = y_pred_eval_both[ np.arange(y_pred_eval_both.shape[0]), self.snpDF.label_numeric_aa_alt.values ]

        y_df = pd.DataFrame({"id": self.snpDF.snp_id, "act": self.y_ddd[:, 0], "pred": alt_score, "predJigsaw": alt_score_jigsaw, "predBoth": alt_score_both})

        sig = ranksums(y_df[ y_df.act == 1.0 ]["pred"], 
                       y_df[ y_df.act == 0.0 ]["pred"]).pvalue
        sigJigsaw = ranksums(y_df[ y_df.act == 1.0 ]["predJigsaw"], 
                       y_df[ y_df.act == 0.0 ]["predJigsaw"]).pvalue
        sigBoth = ranksums(y_df[ y_df.act == 1.0 ]["predBoth"], 
                       y_df[ y_df.act == 0.0 ]["predBoth"]).pvalue
        
        sigDict = { "sig_score_raw_alt_%s" % self.evalName: np.log10(sig),
                    "sig_score_rawJigsaw_alt_%s" % self.evalName: np.log10(sigJigsaw),
                    "sig_score_rawBoth_alt_%s" % self.evalName: np.log10(sigBoth) }

        print(sigDict)

        if savePredFile != None:
            y_df.to_csv(savePredFile)

        print("Done ranksum eval (%s)" % str( time.time() - t ))
        
        return y_df, sigDict



    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        y_df, sigDict = self.performEval(self.model)

        for sigName, sigVal in sorted(sigDict.items(), key=lambda x: x[0]):
            logs[sigName] = sigVal
            self.history.setdefault(sigName, []).append(sigVal)
        
        
        
