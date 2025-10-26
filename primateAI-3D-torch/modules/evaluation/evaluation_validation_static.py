import os

import numpy as np
from sklearn.metrics import roc_auc_score
import time

from helper.helper_data import getVariantTuples_fromDF, readSnpFilePath
from helper.helper_evaluation import posDictsToDF

SAMPLE_TYPE_PAI = 0
SAMPLE_TYPE_JIGSAW = 1

def sampleTypeToNr(sampleType):
    if sampleType == "pai":
        return SAMPLE_TYPE_PAI
    elif sampleType == "jigsaw":
        return SAMPLE_TYPE_JIGSAW
    else:
        raise Exception("unknown sample type")



class EvaluationValidationStatic():
    def __init__(self, c, valFilePathTuples, evalPosCollection):
        self.c = c
        #self.tboard = tboard
        self.evalName = "eval_validation"
        self.savePredFilePrefix = os.path.join(c["runFolder"], "eval", self.evalName)
        self.evalPosCollection = evalPosCollection


        self.posDictsSorted = []
        #snpDFs = []

        for valName, valFilePath in valFilePathTuples:
            snpDF = readSnpFilePath(valFilePath)
            # snpDF["sample_type"] = sampleTypeToNr(valName)
            # snpDFs.append(snpDF)
            #

            posDicts_local = getVariantTuples_fromDF(snpDF, c['mask_value'])
            for posDict in posDicts_local:
                posDict["sample_type"] = sampleTypeToNr(valName)
                self.posDictsSorted.append(posDict)

        #self.snpDF = pd.concat(snpDFs)
        #self.posDF = self.snpDF[["gene_name", "change_position_1based"]].drop_duplicates()


        self.posDictsSorted = list(sorted(self.posDictsSorted, key=lambda x: x["gene_name"]))

        self.evalPosCollection.addPositions_dicts(self.posDictsSorted)


    def evaluate(self, epoch):

        t = time.time()



        #snpDF_withScores = self.snpDF.merge(posDF_sorted, on=["gene_name", "change_position_1based"])


        #outputScores_jigsaw = scoreDict["scores_jigsaw"]
        #outputScores_full = scoreDict["scores_full"]

        labels_list = []
        sampleTypes = []
        for posDict in self.posDictsSorted:
            labels_list.append( posDict["labelArr"][np.newaxis, :] )
            sampleTypes.append(posDict["sample_type"])

        evalCollection_posDF = self.evalPosCollection.posDF_sorted
        posDF = posDictsToDF( self.posDictsSorted )
        posDF_withScores = posDF.merge(evalCollection_posDF, on=["gene_name", "change_position_1based"])
        assert len(posDF) == len(posDF_withScores)

        #np.array(pd.DataFrame({"test": torch.concat([t1, t2]).numpy().tolist()})["test"].tolist())

        outputScores_jigsaw = np.array(posDF_withScores["scores_jigsaw"].tolist())
        outputScores_full = np.array(posDF_withScores["scores_full"].tolist())


        labels = np.concatenate(labels_list)
        sampleTypes = np.array(sampleTypes)

        # print("static eval I")
        # print("outputScores_jigsaw", outputScores_jigsaw.shape,
        #       "outputScores_full", outputScores_full.shape,
        #       "labels", labels.shape,
        #       "sampleTypes", sampleTypes.shape)

        metricDict = {}
        aucDataDict = {}

        for namei, sampleType, outputScores in [#("jigsaw", SAMPLE_TYPE_JIGSAW, outputScores_jigsaw) ,
                                               ("pai", SAMPLE_TYPE_PAI, outputScores_full)]:

            #print(namei)

            #print(sampleTypes, sampleType)

            sampleTypeIdx = np.where(sampleTypes == sampleType, True, False)

            #print(sampleTypeIdx.shape[0])

            outputScores_local = outputScores[sampleTypeIdx,:]
            labels_local = labels[sampleTypeIdx, :]

            #print("static eval")
            #print(outputScores_local.shape)
            #print(labels_local.shape)

            labels_local_binary = labels_local.copy()
            labels_local_binary = np.where((labels_local_binary >= 0.5) & (labels_local_binary < 1.0001), 1, labels_local_binary)
            labels_local_binary = np.where((labels_local_binary < 0.5) & (labels_local_binary >= 0.0), 0, labels_local_binary)

            outputScores_local_flat = outputScores_local.flatten()
            labels_local_flat = labels_local.flatten()
            labels_local_binary_flat = labels_local_binary.flatten()

            labels_local_flat_goodIdxs = np.where(np.not_equal(labels_local_flat, -1000.0))[0]
            #print(labels_local_flat_goodIdxs.shape[0])

            outputScores_local_flat_good = outputScores_local_flat[labels_local_flat_goodIdxs]
            labels_local_binary_flat_good = labels_local_binary_flat[labels_local_flat_goodIdxs]

            metricDict["auc_" + namei] = roc_auc_score(labels_local_binary_flat_good, outputScores_local_flat_good)

            aucDataDict[namei] = (outputScores_local_flat_good, labels_local_binary_flat_good)


        #metricDict["auc_mean"] = (metricDict["auc_pai"] + metricDict["auc_jigsaw"]) / 2.0

        #outputScores_local_flat_good = np.concatenate([aucDataDict["pai"][0], aucDataDict["jigsaw"][0]])
        #labels_local_binary_flat_good = np.concatenate([aucDataDict["pai"][1], aucDataDict["jigsaw"][1]])

        #print(labels_local_binary_flat_good.shape)

        #metricDict["auc_plain"] = roc_auc_score(labels_local_binary_flat_good, outputScores_local_flat_good)

        # if self.tboard != None:
        #     for metric, auc in metricDict.items():
        #         self.tboard.add_scalar(metric, auc, epoch)

        print("Done static val eval (%s)" % str(time.time() - t))

        return metricDict


