import torch
import collections

mask_val = -1000

def mergeSubprotScoresAndLabels(scoreCollection):
    subprotOutput_toMerge = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(list)))

    for i, subprotOutput in enumerate(scoreCollection.subprot_outputs):

        for key_dnaOrProt, dict_dnaOrProt in subprotOutput.items():
            scoreDict = dict_dnaOrProt["score"]
            labelDict = dict_dnaOrProt["label"]

            for scoreName, scoreTensor in scoreDict.items():
                subprotOutput_toMerge[key_dnaOrProt]["score"][scoreName].append(scoreTensor.clone())

            for labelName, labelTensor in labelDict.items():
                subprotOutput_toMerge[key_dnaOrProt]["label"][labelName].append(labelTensor.clone().detach())

    subprotOutput_merged = collections.defaultdict(lambda: collections.defaultdict(dict))

    for key_dnaOrProt, dict_dnaOrProt in subprotOutput_toMerge.items():
        for key_scoreOrLabel, dict_scoreOrLabel in dict_dnaOrProt.items():
            for keyi, tensorlist in dict_scoreOrLabel.items():
                subprotOutput_merged[key_dnaOrProt][key_scoreOrLabel][keyi] = torch.concat(tensorlist)


    return subprotOutput_merged


class LossCollection():

    def __init__(self):
        self.lossListDict = collections.defaultdict(list)
        self.oldDicts = []

    def reset(self):
        self.oldDicts.append( self.lossListDict )
        self.lossListDict = collections.defaultdict(list)

    def addDict(self, lossDict):
        for lossName, lossVal in lossDict.items():
            self.lossListDict[ lossName ].append(lossVal.item())

    def toTupleList(self):
        tupleList = []
        for lossName, losses in sorted(self.lossListDict.items(), key=lambda x: x[0]):
            lossMean = sum(losses)  / len(losses)
            tupleList.append( (lossName, lossMean) )

        return tupleList

    def toString(self):
        tupleList = self.toTupleList()

        tokens = []
        for lossName, lossVal in tupleList:
            if lossName.startswith("loss"):
                tokens.append( "%s:%.5f" % (lossName, lossVal))

        retString = "; ".join(tokens)

        return retString





