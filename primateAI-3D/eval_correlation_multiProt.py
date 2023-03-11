from nn_worker_helper import getVoxelsForSnpRows_tripleBased_orderSafe, getDsScores, loadDsFile, addScoresToDsDF, convertDsDfToRows
from globalVars import SAMPLE_JIGSAW

from keras.callbacks import Callback

from sklearn.metrics import r2_score

from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import os
import warnings
import time

class MultiprotCorrelationEval(Callback):
    def __init__(self, c, variantsDfFilePath, pdbLmdb, dsDF, verbose=0):

        super(MultiprotCorrelationEval, self).__init__()

        print(variantsDfFilePath)

        if variantsDfFilePath.endswith(".pkl"):
            snpDF = pd.read_pickle(variantsDfFilePath)
        else:
            snpDF = pd.read_csv(variantsDfFilePath)

        if not "phenotype" in snpDF.columns:
            snpDF["phenotype"] = "validation"

        targetColumnsSet = set(["assay_score", "probs_patho", "dsScore", "mean_phenotype"])
        availColsSet = set(snpDF.columns.tolist())

        commonCols = targetColumnsSet & availColsSet
        if len(commonCols) != 1:
            raise Exception("Target col not right:" , str(commonCols))
        targetCol = commonCols.pop()


        sampleDF = loadDsFile(variantsDfFilePath, targetColumns=[targetCol])
        snpRows = convertDsDfToRows(sampleDF, targetCol, c["binaryDsLabels"], constantToAdd=0.0, maxSamples=c["maxSnpSamples"])


        snpRows_jigsaw = []
        for name, change_position_1based, label_numeric_aa, labelArr, sampleType in snpRows:
            snpRows_jigsaw.append((name, change_position_1based, label_numeric_aa, labelArr, SAMPLE_JIGSAW))

        if not dsDF is None:
            raise Exception("Not implemented")


        X_eval, y_eval, countTuples, _ = getVoxelsForSnpRows_tripleBased_orderSafe(snpRows, c, pdbLmdb)
        X_eval_jigsaw, y_eval_jigsaw, countTuples, _ = getVoxelsForSnpRows_tripleBased_orderSafe(snpRows_jigsaw, c, pdbLmdb)

        self.y_eval = y_eval

        self.evalName = os.path.basename(variantsDfFilePath).replace(".csv", "").replace(".pkl", "").replace("_sample", "")
        
        self.X_eval = X_eval
        self.X_eval_jigsaw = X_eval_jigsaw
        self.targetCol = targetCol
        self.snpDF = snpDF

        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.snpRows = snpRows
        self.snpRows_jigsaw = snpRows_jigsaw


    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}


    def performEval(self, model, savePredFile=None):
        t=time.time()

        print("Starting corr eval")

        y_pred = model.predict(self.X_eval)
        y_pred_jigsaw = model.predict(self.X_eval_jigsaw)
        y_pred_both = np.mean([y_pred, y_pred_jigsaw], axis=0)

        dfRows = []
        for predArrName, predArr in [("y_pred_pai", y_pred), ("y_pred_jigsaw", y_pred_jigsaw), ("y_pred_both", y_pred_both)]:
            for rowIdx, (name, change_position_1based, label_numeric_aa, labelArr, sampleType) in enumerate(self.snpRows):
                for label_numeric_aa_alt, label_numeric_func_ref in enumerate(labelArr.tolist()):
                    if label_numeric_func_ref > -900:
                        pred_val = predArr[rowIdx, label_numeric_aa_alt]
                        dfRows.append((name, change_position_1based, label_numeric_aa, label_numeric_aa_alt, label_numeric_func_ref, predArrName, pred_val))

        evalDF = pd.DataFrame(dfRows, columns=["gene_name", "change_position_1based", "label_numeric_aa", "label_numeric_aa_alt", "label_numeric_func_ref", "predArrName", "pred_val"])

        snpDF_merged = self.snpDF.merge(evalDF[["gene_name", "change_position_1based", "label_numeric_aa_alt", "predArrName", "pred_val"]], on=["gene_name", "change_position_1based", "label_numeric_aa_alt"], how="left")

        evalRows = []
        for (phenotype, gene_name, predArrName), grpDF in snpDF_merged.groupby(["phenotype", "gene_name", "predArrName"]):

            if grpDF.pred_val.unique().shape[0] > 5:

                corr, sig = spearmanr(grpDF[self.targetCol], grpDF.pred_val)
                r2 = corr * corr
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for statName, stat in [("corr", np.float64(corr)),
                                           ("sig", -np.log10(sig)),
                                           ("r2", np.float64(r2))]:
                        evalRows.append((phenotype, gene_name, predArrName, statName, stat))


        evalRowsDF = pd.DataFrame(evalRows, columns=["phenotype", "gene_name", "predArrName", "statName", "stat"])
        evalRowsDF = evalRowsDF[np.isfinite(evalRowsDF.stat)]

        evalRowsDF_grpd = evalRowsDF[["predArrName", "statName", "stat"]].groupby(["predArrName", "statName"]).agg(np.mean).reset_index()
        evalRowsDF_grpd["keyi"] = evalRowsDF_grpd["statName"] + "_" + evalRowsDF_grpd["predArrName"].str.replace("y_pred_", "") + "_" + self.evalName

        print(evalRowsDF_grpd)

        sigDict = {}
        for keyi, stat in evalRowsDF_grpd[["keyi", "stat"]].values.tolist():
            sigDict[keyi] = np.float64(stat)

        print(sigDict)

        if savePredFile != None:
            evalDF.to_csv(savePredFile)

        print("Done corr eval (%s)" % str( time.time() - t ))
        
        return evalDF, sigDict


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        print("here2")
        y_df, sigDict = self.performEval(self.model)

        print(sigDict)
        for sigName, sigVal in sorted(sigDict.items(), key=lambda x: x[0]):
            logs[sigName] = sigVal
            self.history.setdefault(sigName, []).append(sigVal)


