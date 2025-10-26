import pandas as pd
import numpy as np
import pickle
import os
import json
from pathlib import Path


def readConfig(configFilePath, datafolder, runfolder):
    print("config", configFilePath)
    c = json.load(open(configFilePath.replace("<dataFolder>", datafolder).replace("<runFolder>", runfolder)))
    for k, v in c.items():
        if isinstance(v, str):
            c[k] = v.replace("<dataFolder>", datafolder).replace("<runFolder>", runfolder)
        elif isinstance(v, list) and len(v) >0:
            newL = []
            for vi in v:
                if isinstance(vi, str):
                    newL.append( vi.replace("<dataFolder>", datafolder).replace("<runFolder>", runfolder) )
                else:
                    newL.append(vi)
            c[k] = newL
    c["runFolder"] = runfolder

    print("runFolder", runfolder)

    return c

def compareDicts(dict1, dict2):
    changed_fields = {}

    for key in dict1:
        if key in dict2 and dict1[key] != dict2[key]:
            changed_fields[key] = {'from': dict1[key], 'to': dict2[key]}

    return changed_fields

def addLabels_singleScore(pdbRepo, genePosDicts, labelName):

    labelSize = next(iter(genePosDicts.values()))[0]["labelArr"].shape[0]

    for i, (geneName, geneDict) in enumerate(pdbRepo.items()):

        if i % 1000 == 0: print(i)

        labels = np.full((geneDict["caAllAtomIndices"].shape[0], labelSize), -1000.0, dtype=np.float32)

        if geneName in genePosDicts:
            posDictsGene = genePosDicts[ geneName ]
            for posDict in posDictsGene:
                labels[ posDict["change_position_1based"], :] = posDict["labelArr"]

        geneDict[labelName] = labels

    return pdbRepo



def addRefAALabels(pdbRepo, mask_value):

    for i, (geneName, geneDict) in enumerate(pdbRepo.items()):
        if i % 1000 == 0: print(i)

        targetAtomIdx = geneDict["caAllAtomIndices"][1:].squeeze().astype("int")
        allRefAAsNum = geneDict["atom_resnamenum"][targetAtomIdx]

        labels = np.full((geneDict["caAllAtomIndices"].shape[0], 20), 0, dtype=np.float32)
        labels[0,:] = mask_value
        labels[ np.arange(1, labels.shape[0]), allRefAAsNum ] = 1

        geneDict["label_refAA"] = labels

    return pdbRepo


def getTooLongGenes(pdbDict, c):
    badGenes = set([])
    for geneNameExt in pdbDict.keys():
        if "_" in geneNameExt:

            splitNr = int(geneNameExt.split("_")[1])

            if splitNr > c["input_maxProtSplitNr"]:
                badGenes.add(geneNameExt.split("_")[0])

    for geneNameExt in pdbDict.keys():
        if "_" in geneNameExt:

            geneName = geneNameExt.split("_")[0]

            if geneName in badGenes:
                badGenes.add(geneNameExt)

    return badGenes


def getTargetFeatures(c):
    paraList= []
    for configKey in c["input_featureKeys"]:
        paraList.extend( c[configKey] )

    targetFeatures = list(sorted(list(set(paraList))))

    return targetFeatures


def createFeatureToFolderMapping(c, targetFeatures):
    """
    Loop through the repo folders (following the order from input_pdbRepoFolder) and
    create a map featureName-->featureFilePath.
    Features that are present in more than one folder should be overwritten when loading (like for configs)
    """
    mappingFeatToFile = {}
    targetFeaturesSet = set(targetFeatures)

    for pdbDictFolderPath in c["input_pdbRepoFolder"]:
        for path, subdirs, files in os.walk(pdbDictFolderPath):
            for name in files:
                absPath = os.path.join(path, name)
                relPath = os.path.relpath(absPath, start=pdbDictFolderPath)
                featureName = relPath.replace("/", "_").replace(".pkl", "")

                if featureName in targetFeaturesSet:
                    mappingFeatToFile[featureName] = absPath

    featuresMissing = targetFeaturesSet - set(mappingFeatToFile.keys())

    assert len(featuresMissing) == 0, "Features missing: %s" % str(featuresMissing)

    return mappingFeatToFile


def loadPdbRepo(c):
    targetFeatures = getTargetFeatures(c)
    mappingFeatToFile = createFeatureToFolderMapping(c, targetFeatures)

    #print( mappingFeatToFile )

    baseFeatName = "change_position_1based" # use change_position_1based from pdbDictFolderPath
    baseFeatFilePath = mappingFeatToFile[ baseFeatName ]
    basePdbDict = pickle.load(open(baseFeatFilePath, "rb"))

    geneNames = basePdbDict.keys()

    geneNames = [g for g in geneNames if not g in ["ENST00000360839.7_4"]]

    pdbDict = {}
    for geneName in geneNames:
        pdbDict[geneName] = DictWrapper()

    rows = []

    randomGeneName = next(iter(basePdbDict.keys()))

    for featName in targetFeatures:
        print("Loading", featName)
        featDict = pickle.load(open(mappingFeatToFile[featName], "rb"))

        for i, geneName in enumerate(geneNames):
            pdbDict[geneName][featName] = featDict[geneName]
        geneArr = featDict[randomGeneName]
        rows.append( (featName, str(geneArr.shape), str(geneArr.dtype), mappingFeatToFile[featName]) )

    for geneName in geneNames:
        pdbDict[geneName].usedKeys = set([])

    print(pd.DataFrame(rows, columns=["Feature name", "Shape", "Dtype", "File path"]))

    return pdbDict


def prepareValSnpDF(c):
    if c["eval_staticValPaiFilePath_benign"].endswith(".csv"):
        benignDataFilePath = pd.read_csv(c["eval_staticValPaiFilePath_benign"], index_col=0)
    else:
        benignDataFilePath = pd.read_pickle(c["eval_staticValPaiFilePath_benign"])  # , index_col=0)

    if c["eval_staticValPaiFilePath_patho"].endswith(".csv"):
        pathoDataFilePath = pd.read_csv(c["eval_staticValPaiFilePath_patho"], index_col=0)
    else:
        pathoDataFilePath = pd.read_pickle(c["eval_staticValPaiFilePath_patho"])  # , index_col=0)

    df = pd.concat([benignDataFilePath, pathoDataFilePath])
    outputFilePath = os.path.join(c["tmpFolder"], "valData.pkl")

    df.to_pickle(outputFilePath)

    return outputFilePath


from collections.abc import MutableMapping
class DictWrapper(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        self.usedKeys = set([])

    def __getitem__(self, key):
        self.usedKeys.add(key)
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key
