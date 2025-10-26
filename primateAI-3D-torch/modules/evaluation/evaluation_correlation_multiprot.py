from helper.helper_file import mkdir_p
from helper.helper_data import getVariantTuples_fromDF, readSnpFilePath, expandRowDF
import os
import pandas as pd
from scipy.stats import ranksums, spearmanr
import numpy as np
import time
import warnings



class EvaluationCorrelationMultiprot():
    def __init__(self, c, variantsFilePath, evalPosCollection):
        self.c = c
        #self.tboard = tboard
        self.variantsFilePath = variantsFilePath
        self.evalPosCollection = evalPosCollection

        snpDF = readSnpFilePath( variantsFilePath)

        if not "phenotype" in snpDF.columns:
            snpDF["phenotype"] = "validation"

        if not "label_numeric_func" in snpDF.columns:
            snpDF["label_numeric_func"] = -1

        targetColumnsSet = {"assay_score", "probs_patho", "dsScore", "mean_phenotype"}
        availColsSet = set(snpDF.columns.tolist())

        commonCols = targetColumnsSet & availColsSet
        if len(commonCols) != 1:
            raise Exception("Target col not right:" , str(commonCols))
        targetCol = commonCols.pop()

        self.posDF = getVariantTuples_fromDF(snpDF, c['mask_value'], asDF=True)

        self.evalName = os.path.basename(variantsFilePath).replace(".csv", "").replace(".pkl", "").replace("_sample", "").replace(".fmt", "")
        self.savePredFilePrefix = os.path.join(c["runFolder"], "eval", self.evalName)

        #self.geneToGeneDicts = model.voxelLayer.voxelizeVariantTuples(posTuples)
        self.snpDF = snpDF

        self.targetCol = targetCol

        self.evalPosCollection.addPositions_df(self.posDF)


    def evaluate(self, epoch):

        t = time.time()



        evalCollection_posDF = self.evalPosCollection.posDF_sorted

        posDF_withScores = self.posDF.merge(evalCollection_posDF, on=["gene_name", "change_position_1based"])

        assert len(self.posDF) == len(posDF_withScores)

        #print(posDF_withScores.columns.tolist())
        scoreCols = [c for c in posDF_withScores.columns.tolist() if c.startswith("score")]
        #print(scoreCols)
        posDF_withScores_exp = expandRowDF(posDF_withScores, scoreCols=scoreCols)


        snpDF_merged = self.snpDF.merge(posDF_withScores_exp[["gene_name", "change_position_1based", "label_numeric_aa_alt", "scoreName", "score"]], 
                                        on=["gene_name", "change_position_1based", "label_numeric_aa_alt"], how="left")

        assert len(snpDF_merged) == len(self.snpDF)*len(scoreCols)


        evalRows = []
        skipped = []
        for (phenotype, gene_name, scoreName), grpDF in snpDF_merged.groupby(["phenotype", "gene_name", "scoreName"]):
            # print(gene_name)

            if grpDF.score.unique().shape[0] > 5 and grpDF[self.targetCol].unique().shape[0] > 1:

                corr, sig = spearmanr(grpDF[self.targetCol], grpDF.score)
                r2 = corr * corr  # r2_score(grpDF[self.targetCol], grpDF.score)

                # print(corr, sig, r2, -np.log10(sig))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for statName, stat in [("corr", np.abs(np.float64(corr))),
                                           ("signed_corr(debug)", np.float64(corr)),
                                           ("sig", -np.log10(sig)),
                                           ("r2", np.float64(r2))]:
                        evalRows.append((phenotype, gene_name, scoreName, statName, stat))

            else:
                skipped.append( (phenotype, gene_name, scoreName, grpDF.score.unique().shape[0], grpDF[self.targetCol].unique().shape[0]) )


        evalRowsDF = pd.DataFrame(evalRows, columns=["phenotype", "gene_name", "scoreName", "statName", "stat"])
        evalRowsDF = evalRowsDF[np.isfinite(evalRowsDF.stat)]

        evalRowsDF_grpd_mean = evalRowsDF[["scoreName", "statName", "stat"]].groupby(["scoreName", "statName"]).agg(np.mean).reset_index()
        evalRowsDF_grpd_mean["statName"] = evalRowsDF_grpd_mean["statName"] + "_mean"

        evalRowsDF_grpd_median = evalRowsDF[["scoreName", "statName", "stat"]].groupby(["scoreName", "statName"]).agg(np.median).reset_index()
        evalRowsDF_grpd_median["statName"] = evalRowsDF_grpd_median["statName"] + "_median"

        evalRowsDF_grpd = pd.concat([evalRowsDF_grpd_mean, evalRowsDF_grpd_median])

        evalRowsDF_grpd["keyi"] = evalRowsDF_grpd["statName"] + "_" + evalRowsDF_grpd["scoreName"] + "_" + self.evalName

        #print(evalRowsDF_grpd)

        metricDict = {}
        for keyi, stat in evalRowsDF_grpd[["keyi", "stat"]].values.tolist():
            # if self.tboard != None:
            #     self.tboard.add_scalar(keyi, stat, epoch)
            metricDict[keyi] = np.float64(stat)


        if self.c['eval_logAllEvalScores']:
            savePredFile = "%s.epoch%03d.pkl.gz" % (self.savePredFilePrefix, int(epoch))

            mkdir_p(os.path.dirname(savePredFile))
            snpDF_merged.to_pickle(savePredFile, compression="gzip")

        print("Done corr eval (%s)" % str(time.time() - t))


        return metricDict

