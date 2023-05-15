
import numpy as np
import tensorflow as tf
import keras.backend as K
from multiprocessing import Value

import pandas as pd

from multiprocessing import Process, Queue
import os
import lmdb
import pickle
import glob

from scipy.stats import special_ortho_group
from keras.utils import Sequence, np_utils

import globalVars
from globalVars import SAMPLE_JIGSAW, SAMPLE_DS
from voxelization_accel import  voxelize_triples
import time
from _collections import defaultdict
import random
import traceback

from voxelization_accel_helpers import getGridCenters, getCenterNeighbors, voxelizeFromTriples, getNFeats, voxelizeFromTriples_idxList
from nn_worker_multizhelper import getVoxelGridToNNMap
from fileOps import mkdir_p, touchFile
import shutil

import collections


def chunks(l, n):
    chunksRet = []
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        chunksRet.append(l[i:i + n])
    return chunksRet


def initRotMatrices(c):
    rotMatrices = []
    if c["rotate"]:
        for i in range(10000):
            rotMat = np.float32(special_ortho_group.rvs(3))
            rotMatrices.append(rotMat)
    return np.array(rotMatrices)


def getPaiRows(trainSnpFilePath, dsDF, sampleType=0, allowMissingLabel=False, maxSamples=-1): #
    print(trainSnpFilePath)

    if trainSnpFilePath.endswith(".csv"):
        paiDF = pd.read_csv(trainSnpFilePath, index_col=0)
    else:
        paiDF = pd.read_pickle(trainSnpFilePath) #, index_col=0)

    if not "label_numeric_func" in paiDF.columns:
        print("Adding label_numeric_func")
        paiDF["label_numeric_func"] = np.where(paiDF["label_pai"] == "benign", 0, 1)

    if not dsDF is None:
        print("Reducing to DS variants")
        print("Before: ", len(paiDF))
        paiDF = paiDF.merge(dsDF, on=dsDF.columns.tolist())
        print("After: ", len(paiDF))

    if (not "label_numeric_func" in paiDF) and not allowMissingLabel:
        raise Exception("NO!")
        # paiDF["label_numeric_func"] = np.where(paiDF["label"] == "benign", 0, 1)
        # paiDF["label_numeric_aa"] = np.array([ aaCharToNum(aa) for aa in paiDF.aa_pdb ])
        # paiDF["label_numeric_aa_alt"] = np.array([ aaCharToNum(aa) for aa in paiDF.alt_aa ])

    paiDF_tmp = paiDF[["name", "change_position_1based", "label_numeric_aa", "label_numeric_aa_alt", "label_numeric_func"]].dropna() #
    paiDF_tmp["change_position_1based"] = paiDF_tmp["change_position_1based"].astype("int")
    paiDF_tmp["label_numeric_func"] = paiDF_tmp["label_numeric_func"].astype("int")
    paiDF_tmp["label_numeric_aa"] = paiDF_tmp["label_numeric_aa"].astype("int")
    paiDF_tmp["label_numeric_aa_alt"] = paiDF_tmp["label_numeric_aa_alt"].astype("int")

    return getPaiRows_helper(paiDF_tmp, sampleType=sampleType, maxSamples=maxSamples)

def getPaiRows_helper(paiDF_tmp, sampleType=0, maxSamples=-1): #

    print(paiDF_tmp)
    print(len( paiDF_tmp ))

    paiRows = []

    if maxSamples > 0:
        paiDF_tmp = paiDF_tmp.copy()
        paiDF_tmp = paiDF_tmp.sample(frac=1.0).reset_index(drop=True)
        paiDF_tmp = paiDF_tmp.head(maxSamples)


    tmpDict = collections.defaultdict(list)

    for name, change_position_1based, label_numeric_aa, label_numeric_aa_alt, label_numeric_func in paiDF_tmp[["name",
                                                                                                               "change_position_1based",
                                                                                                               "label_numeric_aa",
                                                                                                               "label_numeric_aa_alt",
                                                                                                               "label_numeric_func"]].values.tolist():
        tmpDict[ (name, change_position_1based, label_numeric_aa) ].append( (label_numeric_aa_alt, label_numeric_func) )

    for (name, change_position_1based, label_numeric_aa), varList in tmpDict.items():
        labelArr = np.full(20, -1000.0, dtype=np.float32)

        for label_numeric_aa_alt, label_numeric_func in varList:

            labelArr[ int(label_numeric_aa_alt) ] = label_numeric_func

        paiRows.append( (name, change_position_1based, label_numeric_aa, labelArr, sampleType) )
    
    random.shuffle(paiRows)

    print(len( paiRows ))

    return paiRows


def getJigsawRows(jigsawTrainFilePath, pdbLmdb, dsDF, paiFormat=False, maxSamples=-1):

    if paiFormat:
        jigsawRows = getPaiRows(jigsawTrainFilePath, dsDF, sampleType=SAMPLE_JIGSAW, maxSamples=maxSamples) #, allowMissingLabel=False

    else:
        raise Exception("Not implemented")
        jigsawDF = pd.read_pickle(jigsawTrainFilePath) #, index_col=0)

        jigsawDict = collections.defaultdict(list)
        for protId, resId, aaNr in jigsawDF.values.tolist():
            jigsawDict[ protId ].append( resId )
        
        jigsawRows, _ = sampleToRows(jigsawDict, len(jigsawDF), pdbLmdb)

    random.shuffle(jigsawRows)

    return jigsawRows


def getDsScores(dsScoresFilePath, dsRankCols, dsNormMethod, doQuantNorm):
    dsScoresDF = pd.read_pickle(dsScoresFilePath)

    if doQuantNorm:
        grpDFs = []
        for gene_name, grpDF in dsScoresDF.dropna(subset=dsRankCols).groupby("gene_name"):
            grpDF_scores = grpDF[dsRankCols].copy()

            grpDF_new = grpDF.copy()

            rank_mean = grpDF_scores.stack().groupby(grpDF_scores.rank(method='first').stack().astype(int)).mean()

            grpDF_scores = grpDF_scores.rank(method='min').stack().astype(int).map(rank_mean).unstack()

            for targetColumn in dsRankCols:
                grpDF_new[targetColumn] = grpDF_scores[targetColumn]

            grpDFs.append(grpDF_new)

        dsScoresDF = pd.concat(grpDFs).reset_index(drop=True)


    print("Using DS normalization", dsNormMethod)

    if dsNormMethod in ["raw", "percentile", "minmax"]:
        dsScoresDF["dsScore"] = dsScoresDF[dsRankCols].mean(axis=1)

    if dsNormMethod == "percentile":
        dsScoresDF["dsScore"] = dsScoresDF["dsScore"].rank(pct=True)

    elif dsNormMethod == "minmax":
        dsScoresDF["dsScore"] = (dsScoresDF["dsScore"] - dsScoresDF["dsScore"].min()) / (dsScoresDF["dsScore"].max() - dsScoresDF["dsScore"].min())

    elif dsNormMethod == "rank":
        dsScoresDF["dsScore"] = dsScoresDF[dsRankCols].rank().mean(axis=1) / len(dsScoresDF)

    if "probs_patho" in dsScoresDF.columns.tolist():
        dsScoresDF = dsScoresDF.drop(columns = ["probs_patho"])  # ["varScore_classifier", "varScore_language", "pred_score", "probs_patho_ds", "probs_patho_language"])

    return dsScoresDF

def loadDsFile(dsTrainFilePath, targetColumns = ["probs_patho"]):
    if dsTrainFilePath.endswith(".pkl"):
        dsSampleDF = pd.read_pickle(dsTrainFilePath).copy() #, "ref_aa", "alt_aa"
    else:
        dsSampleDF = pd.read_csv(dsTrainFilePath).copy() #, "ref_aa", "alt_aa"

    if not "gene_name" in dsSampleDF:
        dsSampleDF["gene_name"] = dsSampleDF["name"]
        dsSampleDF = dsSampleDF.drop(columns="name")

    dsSampleDF = dsSampleDF[["gene_name", "change_position_1based", "label_numeric_aa", "label_numeric_aa_alt"] + targetColumns].copy()

    return dsSampleDF


def addScoresToDsDF(dsSampleDF, dsScoreDF):
    dsSampleDF["isOld"] = True
    dsScoreDF["isNew"] = True

    dsSampleDF["label_numeric_aa"] = dsSampleDF["label_numeric_aa"].astype("int")
    dsSampleDF["label_numeric_aa_alt"] = dsSampleDF["label_numeric_aa_alt"].astype("int")

    dsScoreDF["label_numeric_aa"] = dsScoreDF["label_numeric_aa"].astype("int")
    dsScoreDF["label_numeric_aa_alt"] = dsScoreDF["label_numeric_aa_alt"].astype("int")

    mDF = dsSampleDF.merge(dsScoreDF, on=["gene_name", "change_position_1based", "label_numeric_aa", "label_numeric_aa_alt"], how="outer")

    lostNew = np.sum(mDF.isOld.isna())
    lostOld = np.sum(mDF.isNew.isna())

    mDF = mDF.dropna(subset=["dsScore", "probs_patho"])
    mDF["label_numeric_aa"] = mDF["label_numeric_aa"].astype("int")
    mDF["label_numeric_aa_alt"] = mDF["label_numeric_aa_alt"].astype("int")

    corrs = []
    for gene_name, geneDF in mDF.groupby("gene_name"):
        corr = geneDF[["dsScore", "probs_patho"]].corr(method="spearman").values[0, 1]
        corrs.append(corr)

    corrsSeries = pd.Series(corrs)

    meanCorr = corrsSeries.dropna().mean()

    print("%d DS scores; lost new: %d; lost old: %d; mean corr: %.4f" % (len(mDF), lostNew, lostOld, meanCorr))

    return mDF

def convertDsDfToRows(dsSampleDF, targetColumn, binaryDsLabels, constantToAdd = 10, maxSamples=-1):
    dsSampleDF = dsSampleDF.copy()

    dsSampleDF.rename(columns={"gene_name": "name"}, inplace=True)

    if binaryDsLabels:
        dsSampleDF["label_numeric_func"] = np.where(dsSampleDF[targetColumn] > 0.5, 1+constantToAdd, constantToAdd)
    else:
        dsSampleDF["label_numeric_func"] = dsSampleDF[targetColumn] + constantToAdd

    dsSampleDF.drop(columns=[targetColumn], inplace=True)

    dsRows = getPaiRows_helper(dsSampleDF, sampleType=SAMPLE_DS, maxSamples=maxSamples)

    return dsRows


def encodeLabels(c, snpDF_noNA):
    if c["targetLabel"].endswith("_func"):
        labelsArr = np_utils.to_categorical(snpDF_noNA[c["targetLabel"]].values)
        
    else:
        tmpDF = snpDF_noNA[ c["targetLabel"] ].astype("str").str.split("_", expand=True).astype("int")
        tmpDF.columns = ["aa%d" % i for i in range(len(tmpDF.columns))]
        
        colEncodings = [ np_utils.to_categorical(tmpDF[coli], num_classes=22)[:, np.newaxis, :] for coli in tmpDF.columns ]

        labelsArr = np.concatenate(colEncodings, axis=1)

    return labelsArr


def initLabelEncoding(c, pdbRepoDict):

    evoLabelDict = collections.defaultdict(dict)
    labelsDict = {}

    for i in range(22):
        labelsArr = np_utils.to_categorical(i, num_classes=22)#[:, np.newaxis]
        labelsDict[i] = labelsArr

    return labelsDict, evoLabelDict

#@njit()
def labelOneHot(labelsArrOrig, protPosResnamenum):
    labelsArr = np.where(labelsArrOrig > 0, 1.0, 0.0)
    labelsArr[ protPosResnamenum ] = -1.0

    return labelsArr

def labelOneHotBalanced(labelsArrOrig, protPosResnamenum):
    labelsArr_tmp = labelOneHot(labelsArrOrig, protPosResnamenum)

    onePoss = np.where(labelsArr_tmp == 1.0)
    zeroPoss = np.where(labelsArr_tmp == 0.0)

    minLen = np.minimum( onePoss.shape[0], zeroPoss.shape[0] )

    labelsArr = np.full(labelsArr_tmp.shape, -1.0)

    if minLen > 0:
        oneSample = np.random.choice(onePoss, size=minLen)
        zeroSample = np.random.choice(zeroPoss, size=minLen)

        labelsArr[ oneSample ] = 1.0
        labelsArr[ zeroSample ] = 0.0
        

    return labelsArr


def getSampleWeight(posTuple, pdbRepoDict, c):
    
    protId, protPos, protPosResnamenum, labelsArr, sampleType = posTuple

    if sampleType == SAMPLE_JIGSAW:
        return c["jigsawSampleWeight"]
    elif sampleType == SAMPLE_DS:
        return 1.0
    else:
        return 1.0

def getLabelArrForPos(posTuple, pdbRepoDict, c):
    protId, protPos, protPosResnamenum, labelsArr, isJigsaw = posTuple


    return labelsArr



def loadAaEncoding():
    aaEncodedDict = {}

    for i in range(20):
        refAaEncoded = np_utils.to_categorical(np.array([i]), num_classes=20)
        
        for j in range(20):
            altAaEncoded = np_utils.to_categorical(np.array([j]), num_classes=20)
            
            aaEncoded = np.concatenate([refAaEncoded, altAaEncoded], axis=1)[0]
            
            aaEncodedDict[ (i, j) ] = aaEncoded

    return aaEncodedDict

def loadValRows(c):
    valTuples = []

    pdbRepoDict = globalVars.globalVars["pdbRepoDict"]

    nValProts = int(len(pdbRepoDict.items()) * c["trainTestFraction"])
    nValSamples = c["nValSamples"]

    valProtIds = set(random.sample(pdbRepoDict.keys(), k=nValProts))

    protLenDF = initProtLenDF(valProtIds)
    protLenDF_train = initProtLenDF([ protId for protId in pdbRepoDict.keys()  ]) #if not protId in valProtIds

    valSample = getPosSample(protLenDF["protLenCumsum"].values, protLenDF["protId"].values.astype('str'), nValSamples)
    t=time.time()

    print("Done shuffling %s" % str(time.time()-t))

    valTuples, aaCounter = sampleToRows(valSample, nValSamples, pdbRepoDict, balanced=c["balancedAaSample"])
    random.shuffle(valTuples)

    return valTuples, protLenDF_train


def initProtLenDF(protIds):

    protLenRows = []

    pdbRepoDict = globalVars.globalVars["pdbRepoDict"]

    for protId in protIds:
        protDict = pdbRepoDict[protId]
        
        #protDict[protId] = protDict["caIndexArray"].shape[0]

        caIndexs = protDict["caIndexArray"]

        protLenRows.append( (protId, caIndexs.shape[0]) )

    df = pd.DataFrame(protLenRows, columns=["protId", "protLen"])
    df["protLenCumsum"] = df["protLen"].cumsum()
    
    return df

from numba import njit
@njit()
def getPosSample_helper(rowSums, prots, sampleSize, rowSample):
    
    currRowIdx = 0
    currRowSum = rowSums[currRowIdx]
    prevRowSum = 0

    posSamples = []

    prevProt = ""
    for i in range(rowSample.shape[0]):

        currSample = rowSample[i]

        while currSample >= currRowSum:
            currRowIdx += 1
            prevRowSum = currRowSum
            currRowSum = rowSums[currRowIdx]

        currProt = prots[currRowIdx]
        currPos = currSample - prevRowSum
        currProtLen = currRowSum - prevRowSum

        while currPos == 0:
            currPos = random.randint(1, currProtLen - 1)

        posSamples.append( (currProt, currPos) )

    return posSamples


def getPosSample(rowSums, prots, sampleSize, rowSample=None):

    posSampleDict = collections.defaultdict(list)
    if sampleSize > 0:
        if rowSample is None:
            rowSample = np.sort(np.random.randint(0, rowSums[-1], size=int(sampleSize*1.5))  )
        
        posSample = getPosSample_helper(rowSums, prots, int(sampleSize*1.5), rowSample)
        posSampleSet = frozenset(posSample)
        posSample = list(posSampleSet)
        random.shuffle(posSample)
        
        print("Here: ", len(posSample), sampleSize)
        assert( len(posSample) >= sampleSize )
        

        for protId, pos in posSample[:sampleSize]:
            posSampleDict[protId].append(pos)
        
    return posSampleDict


from numba import njit
#@njit()

@njit()
def getEvoPoss_oneProt(protId, evoArray, caIndexArray, resNameNumArray, poss, balanced):
    nRes = poss.shape[0]

    benignPossArray = np.full( (nRes*21, 2), -1, dtype=np.uint16 )
    pathoPossArray  = np.full( (nRes*21, 2), -1, dtype=np.uint16 )
    newEvoArray = np.full( (evoArray.shape[0], 20) , -1000.0)
    
    benignPossArrayIdx = 0
    pathoPossArrayIdx = 0
    
    for resIdIdx in range(nRes):
        resId = poss[ resIdIdx ]
        
        caIndex = int(caIndexArray[resId][0])
        resNameNum = resNameNumArray[caIndex]
        
        for aaNum in range(20):
            if aaNum != resNameNum:
                if evoArray[resId, aaNum] > 0:
                    benignPossArray[ benignPossArrayIdx, 0 ] = resId
                    benignPossArray[ benignPossArrayIdx, 1 ] = aaNum
                    
                    benignPossArrayIdx += 1
                
                else:
                    pathoPossArray[ pathoPossArrayIdx, 0 ] = resId
                    pathoPossArray[ pathoPossArrayIdx, 1 ] = aaNum
                    
                    pathoPossArrayIdx += 1
    
    
    if balanced:
        sampleCountBenign = np.amin(np.array([pathoPossArrayIdx, benignPossArrayIdx]))
        sampleCountPatho = np.amin(np.array([pathoPossArrayIdx, benignPossArrayIdx]))
    else:
        sampleCountBenign = benignPossArrayIdx
        sampleCountPatho = pathoPossArrayIdx

    if sampleCountBenign > 0:
        benignIdxs = np.random.choice( np.arange(0, benignPossArrayIdx), size=sampleCountBenign, replace=False )

        for i in range(sampleCountBenign):
            benignIdx = benignIdxs[i]
            resId = benignPossArray[ benignIdx, 0 ]
            aaNum = benignPossArray[ benignIdx, 1 ]
            
            newEvoArray[resId, aaNum] = 1.0
            
            #print(i, np.unique(newEvoArray, return_counts=True))
            

    if sampleCountPatho > 0:
        pathoIdxs = np.random.choice(np.arange(0, pathoPossArrayIdx), size=sampleCountPatho, replace=False)

        for i in range(sampleCountPatho):
            pathoIdx = pathoIdxs[i]
            resId = pathoPossArray[ pathoIdx, 0 ]
            aaNum = pathoPossArray[ pathoIdx, 1 ]
            
            newEvoArray[resId, aaNum] = 0.0

    returnRows = []
    
    isJigsaw = True

    vals = []
    for resIdIdx in range(nRes):
        resId = poss[ resIdIdx ]
        
        caIndex = int(caIndexArray[resId][0])
        resNameNum = resNameNumArray[caIndex]
        
        labelArr = newEvoArray[resId, :]
        
        for i in range(20):
            if newEvoArray[resId, i] != -1:
              vals.append( newEvoArray[resId, i] )

        returnRows.append( (protId, resId, resNameNum, labelArr, isJigsaw) )
        

    return returnRows
        

def sampleToRows(sample, sampleSize, pdbLmdb, balanced=False, doAssert=True):

    rowTuples = []

    if not sample is None:
        sampleList = list(sample.items())

        print("Converting to sample rows...")

        random.shuffle(sampleList)

        with pdbLmdb.begin() as pdbTxn:

            for i, (currProtId, poss) in enumerate(sampleList):

                possArray = np.array(poss)
                if i % 1000 == 0: print(i)

                protDictBytes = pdbTxn.get(currProtId.encode("ascii"))
                protDict = pickle.loads(protDictBytes)

                evoArray = protDict["evoArray"]
                caIndexArray = protDict["caIndexArray"]
                resNameNumArray = protDict["resnamenum"]

                evoPossSample = getEvoPoss_oneProt(currProtId, evoArray, caIndexArray, resNameNumArray, possArray, balanced)

                rowTuples.extend( evoPossSample )

    random.shuffle(rowTuples)
    
    aaCounter = collections.defaultdict(lambda : 0)
    for rowTuple in rowTuples:
        refAaNum = rowTuple[-3]
        aaCounter[refAaNum] += 1

    if doAssert:
        assert(len(rowTuples) >= sampleSize)
        
    print("Converting to sample rows done.")

    return rowTuples, aaCounter


def getVoxelsForSnpRows_tripleBased_orderSafe(snpRows, c, pdbLmdb):
    
    nrChunks = int(len(snpRows) / 500) + 1
    
    snpRowsChunks = np.array_split(np.array(snpRows, dtype=object), nrChunks)
    
    print("Generated chunks of SNP rows for voxelization")
    
    processes = []
    q = Queue()
    l = []
    maxProcesses = 5

    print("Chunks: ", len(snpRowsChunks), " Chunksize: ", snpRowsChunks[0].shape[0]) 

    #return 

    for i, snpRowsChunk in enumerate(snpRowsChunks):
        while len(processes) >= maxProcesses:
            for j, processi in processes:
                if processi.is_alive():
                    pass

                elif processi.exitcode != 0:

                    processes.remove( (j, processi) )
                    processi.join()
                else:

                    processes.remove( (j, processi) )
                    processi.join()

            while q.qsize() > 0:
                try:
                    l.append(q.get(False))
                except:
                    pass

            time.sleep(0.5)

        print("Starting process")

        p = Process(target=data_generation_wrap_triples, args=(snpRowsChunk, i, c, q, pdbLmdb))
        p.start()
        processes.append( (i, p) )

    while len(processes) > 0:
        for i, processi in processes:
            if processi.is_alive():
                pass
            elif processi.exitcode != 0:
                processes.remove( (i, processi) )
                processi.join()
            else:
                print("\tProcess %d finished" % i)
                processes.remove( (i, processi) )
                processi.join()

        while q.qsize() > 0:
            try:
                l.append(q.get(False))
            except:
                pass

        time.sleep(0.1)


    while q.qsize() > 0:
        try:
            l.append(q.get(False))
        except:
            pass


    X_list = []
    y_list = []

    d=defaultdict(list)
    dy=defaultdict(list)

    l_sorted = sorted(l, key=lambda x: x[0])

    countTuplesAll = []
    sampleWeightsAll = []
    multiz_geneIds_list = []
    multiz_voxelGridNNs_list = []

    for i, returnTuple_compressed in l_sorted:

        returnTuple_decompressed = returnTuple_compressed#pickle.loads(zlib.decompress(returnTuple_compressed))
        
        X_chunk, y_chunk, countTuples, sampleWeights, multiz_geneIds, multiz_voxelGridNNs = returnTuple_decompressed


        if isinstance(X_chunk, list):
            for j, X_chunk_array in enumerate(X_chunk):
                d[j].append(X_chunk_array)
        else:

            d[0].append(X_chunk)

        if isinstance(y_chunk, list):
            for j, y_chunk_array in enumerate(y_chunk):
                dy[j].append(y_chunk_array)
        else:
            dy[0].append(y_chunk)

        sampleWeightsAll.extend(sampleWeights)

        countTuplesAll.extend(countTuples)
        multiz_geneIds_list.append(multiz_geneIds)
        multiz_voxelGridNNs_list.append(multiz_voxelGridNNs)


    for i in range(len(d.keys())):
        X_list.append( np.concatenate(d[i]) )

    for i in range(len(dy.keys())):
        y_list.append( np.concatenate(dy[i]) )


    X_list[2] = np.concatenate([np.array([0]), np.cumsum(X_list[2])])  #tripleLengths
    X_list[5] = np.concatenate([np.array([0]), np.cumsum(X_list[5])])  #tripleLengthsGlobal

    print("Val data shape triples: %s %s" % ( str([X_list[i].shape for i in range(len(X_list))]), 
                                      str([y_list[i].shape for i in range(len(y_list))]) )) #str(X_val[1].shape),  %s


    nFeats = getNFeats( c["nFeatsSeq"],
                len(c["targetAtoms"]),
                c["nFeatsEvo"],
                c["nFeatsAltRef"],
                c["nFeatsAllAtomDist"],
                c["nFeatsProtQual"],                    
                c["includeEvoProfs"],
                c["includeAlt"],
                c["includeAllAtomDist"],
                c["includeProtQual"])


    print("Restoring voxels")
    X = voxelizeFromTriples(X_list[0], 
                            X_list[1], 
                            X_list[2], 
                            X_list[3],
                            X_list[4],
                            X_list[5],
                            nFeats, 
                            np.float32(c["nVoxels"][0])).astype(np.float32)

    print("Val data shape voxels: %s %s" % ( str(X.shape), 
                                      str([y_list[i].shape for i in range(len(y_list))]) )) #str(X_val[1].shape),  %s

    if len(y_list) == 1:
        y = y_list[0]
    else:
        y = y_list

    multiz_geneIds = np.concatenate(multiz_geneIds_list)[:,np.newaxis]
    multiz_voxelGridNNs = np.concatenate(multiz_voxelGridNNs_list)

    if c["doMultiz"]:
        X_list = [X, multiz_geneIds, multiz_voxelGridNNs]
    else:
        X_list=X

    return X_list, y, countTuplesAll, sampleWeightsAll

def loadPdbRepo(fileAbs, evoFeats, scratchDir):
    pdbRepo = pickle.load(open(fileAbs[0], "rb"))

    allKeyNames = list(pdbRepo.values())[0].keys()
    allFeatNames = [k for k in allKeyNames if k.startswith("feat_")]

    badFeats = set(allFeatNames) - set(evoFeats) # + ["feat_prof"]

    dbPathLocal = os.path.join(scratchDir, "pdbLmdb")
    if os.path.exists(dbPathLocal):
        print("Removing", dbPathLocal)
        shutil.rmtree(dbPathLocal)
    mkdir_p(dbPathLocal)
    map_size = 1024 * 1024 * 1024 * 30

    env = lmdb.open(dbPathLocal, map_size=map_size)

    mins = []
    maxs = []
    stds = []

    print("Feats bad:", badFeats, "good:", evoFeats ) # + ["feat_prof"]

    geneNameToId = {}
    idToGeneName = {}

    with env.begin(write=True) as txn:
        for i, (geneName, geneDict) in enumerate(pdbRepo.items()):
            if i % 1000 == 0: print(i)

            geneNameToId[geneName] = i
            idToGeneName[i] = geneName

            featProfZeros = np.zeros( geneDict["feat_prof"].shape, dtype=geneDict["feat_prof"].dtype )

            evoArrs = [featProfZeros]
            for evoFeat in allFeatNames:
                if evoFeat in badFeats:
                    del geneDict[evoFeat]
                else:
                    evoArr = geneDict[evoFeat]

                    evoArrTmp = evoArr.flatten()
                    evoArrTmp = evoArrTmp[ ~np.isnan(evoArrTmp) ]

                    mins.append( (evoFeat, np.min(evoArrTmp) ))
                    maxs.append( (evoFeat, np.max(evoArrTmp) ))
                    stds.append( (evoFeat, np.std(evoArrTmp) ) )

                    evoArrs.append(evoArr)
                    del geneDict[evoFeat]

            evoArr = np.ascontiguousarray(np.concatenate(evoArrs, axis=1)).astype(np.float32)

            if np.isnan(np.sum(evoArr[1:,:].flatten())):
                raise Exception("NaN in input!", geneName)

            geneDict["feat_cons"] = evoArr

            txn.put(geneName.encode("ascii"), pickle.dumps(geneDict))

    if len(mins) > 0:
        print(pd.DataFrame(mins, columns=["feat", "mini"]).groupby("feat").agg(np.min))
        print(pd.DataFrame(maxs, columns=["feat", "maxi"]).groupby("feat").agg(np.max))
        print(pd.DataFrame(stds, columns=["feat", "stdi"]).groupby("feat").agg(np.mean))
    else:
        print("Nothing to aggregate...")

    print("Writing LMDB")
    lmdbObj = lmdb.open(dbPathLocal, create=False, subdir=True, readonly=True, lock=False)

    return lmdbObj, dbPathLocal, mins, maxs, geneNameToId, idToGeneName


def loadMultizDB(c, prefix):

    tmpTmpFolder = os.path.join(c["ramDiskOutputFolderPathBase"], "multizDB")

    if True:
        print("Creating %s" % tmpTmpFolder)
        mkdir_p(tmpTmpFolder)

        dbFiles = glob.glob(c["multizLmdbPath"] + "/*")
        for dbFile in dbFiles:
            destPath_tmp = os.path.join(tmpTmpFolder, os.path.basename(dbFile))

            print("%s ==> %s" % (dbFile, destPath_tmp))
            shutil.copy(dbFile, destPath_tmp)

            print("==")

    print("Multiz folder open:", tmpTmpFolder)

    lmdbObj = lmdb.open(tmpTmpFolder, create=False, subdir=True, readonly=True, lock=False, max_readers = 12000, max_spare_txns = 10)
    lmdbPath = tmpTmpFolder

    return lmdbObj, lmdbPath



def data_generation_wrap_triples(snpRowList_temp, i, config, queuei, pdbLmdb):
    
    returnTuple = data_generation_triples(snpRowList_temp, config, pdbLmdb)

    print("Putting")
    queuei.put( (i, returnTuple) )
    print("Finished putting")
    
    
    return



def data_generation_triples(snpRowList_temp, config, pdbLmdb, epoch=0, counts=False):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

    X = []
    countTuples = []

    nVoxels = np.array(config["nVoxels"])
    boxSize = (nVoxels * config["voxelSize"]).astype("float32")

    edgeLen = ((boxSize[0] / 2) + 4.6)
    centers = getGridCenters(nVoxels, np.array([0, 0, 0]), config["voxelSize"]).astype("float32")

    boxLen = config["nVoxels"][0] * config["voxelSize"]

    voxelSize_local = np.float32(config["voxelSize"])
    boxLen_half = np.float32(boxLen / 2)
    nVoxels_local = int(config["nVoxels"][0])
    maxVoxelIdx = config["nVoxels"][0] - 1

    centerIdxToNeighborCoords, centerIdxToNeighborIdxs, allGood = getCenterNeighbors(centers, voxelSize_local, boxLen_half, nVoxels_local)

    if not allGood:
        raise Exception("This should not happen")

    tripleIdxss = []
    tripleValss = []
    tripleLengths = []
    tripleIdxGlobals = []
    tripleValsGlobals = []
    tripleLengthsGlobals = []

    centerIdxToResIds_list = []


    with pdbLmdb.begin() as pdbTxn:
        for i in range(snpRowList_temp.shape[0]):

            tripleIdxs, tripleVals, tripleIdxGlobal, tripleValsGlobal, countsTuple, centerIdxToResIds = voxelize_triples( pdbTxn,
                                                                    snpRowList_temp[i],
                                                                    centers,
                                                                    edgeLen,
                                                                    config,
                                                                    centerIdxToNeighborCoords,
                                                                    centerIdxToNeighborIdxs,
                                                                    maxVoxelIdx,
                                                                    boxLen_half,
                                                                    voxelSize_local,
                                                                    nVoxels_local)


            tripleIdxss.append(tripleIdxs)
            tripleValss.append(tripleVals)
            tripleLengths.append(tripleIdxs.shape[0])

            tripleIdxGlobals.append(tripleIdxGlobal)
            tripleValsGlobals.append(tripleValsGlobal)
            tripleLengthsGlobals.append(tripleIdxGlobal.shape[0])

            countTuples.append( tuple(list(snpRowList_temp[i]) + list(countsTuple) ) )

            centerIdxToResIds_list.append( (snpRowList_temp[i][0], centerIdxToResIds) )


    tripleIdxAll = np.concatenate(tripleIdxss)
    tripleValsAll = np.concatenate(tripleValss)
    tripleLengthsAll = np.array(tripleLengths)
    tripleIdxGlobalAll = np.concatenate(tripleIdxGlobals)
    tripleValsGlobalAll = np.concatenate(tripleValsGlobals)
    tripleLengthsGlobalAll = np.array(tripleLengthsGlobals)

    #tripleLengthsAll = np.concatenate([np.array([0]), np.cumsum(tripleLengthsAll)])  #tripleLengths  #tripleLengths
    #tripleLengthsGlobalAll = np.concatenate([np.array([0]), np.cumsum(tripleLengthsGlobalAll)])  #tripleLengthsGlobal
    
    X = [tripleIdxAll, 
         tripleValsAll, 
         tripleLengthsAll, 
         tripleIdxGlobalAll,
         tripleValsGlobalAll,
         tripleLengthsGlobalAll]

    y_list = []
    sample_weights = []
    for i in range(snpRowList_temp.shape[0]):
        y_list.append( getLabelArrForPos(snpRowList_temp[i], pdbLmdb, config)  )
        sample_weights.append( getSampleWeight( snpRowList_temp[i], pdbLmdb, config ) )

    y = concatenateLabels(y_list)

    multiz_geneNames, multiz_voxelGridNNs = getVoxelGridToNNMap(centerIdxToResIds_list, config["nVoxels"][0], globalVars.globalVars["geneNameToId"])
    #multiz_varArr, multiz_voxelGridNNs = getMultizData(centerIdxToResIds_list, multizLmdb, config["nVoxels"][0] )

    returnTuple = ( X if len(X) > 1 else X[0], 
                    y if len(y) > 1 else y[0],
                    countTuples,
                    sample_weights,
                    multiz_geneNames,
                    multiz_voxelGridNNs)

    returnTuple_compressed = returnTuple #zlib.compress(pickle.dumps( returnTuple ))

    return returnTuple_compressed





def concatenateLabels(labels):
    firstLabel = labels[0]
    
#     print("First")
#     print(firstLabel)
#     print(firstLabel.shape)
#     print(len(firstLabel.shape))
    
    if firstLabel.shape[0] > 1 and len(firstLabel.shape) > 1:
        newLabels = []
        for dimi in range(firstLabel.shape[0]):
            currLabels = []
            for labeli in labels:
                currLabels.append( labeli[dimi][np.newaxis,:] )
            newLabels.append( np.concatenate(currLabels) )
                        
        return newLabels
    else:
        newLabels = []
        for labeli in labels:
            newLabels.append(labeli[np.newaxis, :])
    
        return [np.concatenate(newLabels)]
    
    

def saveBatchData(outputFolderPathBase, batchTuple, batchIdx, cancel):
    
    X, y, countTuples, sampleWeights, multiz_geneIds, multiz_voxelGridNNs = batchTuple
    X[2] = np.concatenate([np.array([0]), np.cumsum(X[2])])  #tripleLengths  #tripleLengths
    X[5] = np.concatenate([np.array([0]), np.cumsum(X[5])])  #tripleLengthsGlobal

    #outputFolderPreparePath = os.path.join( os.path.dirname(outputFolderPathBase), "prepare", os.path.basename(outputFolderPathBase), "%04d" % batchIdx) 
    #mkdir_p(outputFolderPreparePath)

    outputFolderPath = os.path.join(outputFolderPathBase, "%04d" % batchIdx)
    mkdir_p(outputFolderPath)

    X_names = ["tripleIdxAll", 
         "tripleValsAll", 
         "tripleLengthsAll", 
         "tripleIdxGlobalAll",
         "tripleValsGlobalAll",
         "tripleLengthsGlobalAll"]
    
    for fileName, fileData in zip(X_names, X):
        outputFilePath = os.path.join(outputFolderPath, fileName+".npy")
        np.save(outputFilePath, fileData)
    
    if isinstance(y, list):
        for i, y_i in enumerate(y):
            outputFilePath = os.path.join(outputFolderPath, "y_%d.npy" % i)
            np.save(outputFilePath, y_i)
    else:
        outputFilePath = os.path.join(outputFolderPath, "y.npy")
        np.save(outputFilePath, y)

    for otherFileName, otherData in [("sampleWeights", sampleWeights),
                                     ("multiz_geneIds", multiz_geneIds),
                                     ("multiz_voxelGridNNs", multiz_voxelGridNNs)]:
        outputFilePath = os.path.join(outputFolderPath, "%s.npy" % otherFileName)
        np.save(outputFilePath, otherData)


    touchFilePath = os.path.join(outputFolderPath, ".finished")
    touchFile(touchFilePath)

    if cancel:
        print("Removing ", outputFolderPath)
        shutil.rmtree(outputFolderPath)

    #os.remove(touchFilePath)

    #shutil.move(outputFolderPreparePath, outputFolderPathBase)


def getDsBatches(dsRows, targetNbatches, dsBatchSize):
    print(targetNbatches, " batches with ", len(dsRows), " vars and batch size ", dsBatchSize)

    protToRows = collections.defaultdict(list)
    for rowi in dsRows:
        prot = rowi[0]

        currLen = len(protToRows[prot])

        # print(prot, currLen)

        while currLen == dsBatchSize:
            i = 1
            protNew = prot + ("_%d" % i)
            currLen = len(protToRows[protNew])

            while currLen == dsBatchSize:
                i = i + 1
                protNew = prot + ("_%d" % i)
                currLen = len(protToRows[protNew])

            prot = protNew

        protToRows[prot].append(rowi)

    protToNBatches = []
    for prot, rows in protToRows.items():
        nBatches = len(rows) / float(dsBatchSize)
        protToNBatches.append((prot, nBatches))

    protsSorted = list(sorted(protToNBatches, key=lambda x: x[1], reverse=True))

    nProts = len(set([p for p, n in protsSorted]))
    print("N prots after processing: ", nProts, "; nVars per prot:", len(dsRows) / nProts)

    if len(protsSorted) < targetNbatches:
        raise Exception("Fewer proteins than targeted batches")

    protsSorted_selected = protsSorted[:targetNbatches]

    batchHisto = np.unique(np.array([len(protToRows[p]) for p, b in protsSorted_selected]), return_counts=True)
    print(batchHisto)

    # print(list(sorted(protsSorted, key=lambda x: (x[1], x[0]), reverse=True)))

    batches = []
    for prot, nBatches in protsSorted_selected:
        # print(prot, " --> ", nBatches)

        protVars = protToRows[prot]

        random.shuffle(protVars)
        batches.append(protVars)

    random.shuffle(batches)

    return batches


def getBatches_other(otherSnpRows, targetNbatches, otherBatchSize):
    batches = []

    for i in range(targetNbatches):
        startIndex = i * otherBatchSize

        if i == targetNbatches - 1:
            endIndex = len(otherSnpRows)
        else:
            endIndex = (i + 1) * otherBatchSize

        batchi = otherSnpRows[startIndex:endIndex]

        batches.append(batchi)

    assert (targetNbatches == len(batches))

    return batches

def getBatches(snpRows, c):
    batches = []

    dsRows = [snpRows[i] for i in range(snpRows.shape[0]) if snpRows[i][-1] == SAMPLE_DS]
    otherRows = [snpRows[i] for i in range(snpRows.shape[0]) if snpRows[i][-1] != SAMPLE_DS]
    batchSize = c["batchSize"]

    #nrFullBatchesPossible = int(snpRows.shape[0] / batchSize)

    targetNbatches = int(np.floor(snpRows.shape[0] / batchSize))

    dsRatio = len(dsRows) / float(snpRows.shape[0])

    dsBatchSize = int(np.round(dsRatio * batchSize))
    otherBatchSize = batchSize - dsBatchSize

    dsBatches = [ [] for i in range(targetNbatches) ]
    if dsBatchSize > 0:
        dsBatches = getDsBatches(dsRows, targetNbatches, dsBatchSize)

    otherBatches = getBatches_other(otherRows, targetNbatches, otherBatchSize)
    assert (len(dsBatches) == len(otherBatches))

    print(batchSize, dsBatchSize, otherBatchSize)
    batches = []
    for i, (dsBatch, otherBatch) in enumerate(zip(dsBatches, otherBatches)):
        batchi = dsBatch + otherBatch

        #print(len(dsBatch), len(otherBatch), len(batchi))
        batches.append((i, np.array(batchi, dtype="object") ))

    return batches


def processBatches(pdbLmdb, batchChunk, c, outputFolderBasePath, ignoreCacheLimit, cancel, verbose):
    if verbose: print("Processing batches")
    
    for i, batchi in batchChunk:
        if verbose: print("Batch ", i)
        
        if cancel.value:
            return 

        nrBatchesAvail = len(os.listdir(outputFolderBasePath))
        if verbose: print("Nr batches avail: ", nrBatchesAvail, " / ", c["nrBatchesToCache"])
        
        while nrBatchesAvail >= c["nrBatchesToCache"] and not ignoreCacheLimit and not cancel.value:
            time.sleep(1)
            nrBatchesAvail = len(os.listdir(outputFolderBasePath))
            if verbose: print("Nr batches avail: ", nrBatchesAvail, " / ", c["nrBatchesToCache"])
        

        if cancel.value:
            return 

        if verbose: print("Voxelizing")
        batchData = data_generation_triples(batchi, c, pdbLmdb)


        if cancel.value:
            return 

        if verbose: print("Saving")
        saveBatchData(outputFolderBasePath, batchData, i, cancel.value)
        

def chunks(l, nChunks):
    chunksize = int(round(len(l) / float(nChunks)))

    if len(l) / float(nChunks) < 1:
        return [l]

    maxIdx = len(l)

    currIdx = 0
    
    chunks = []
    while currIdx < maxIdx:
        
        startIdx = len(chunks) * chunksize
        endIdx = (len(chunks)+1) * chunksize
        
        if len(chunks) == nChunks-1:
            endIdx = maxIdx
        
        chunki = l[ startIdx : endIdx ]
        chunks.append(chunki)
        currIdx = endIdx
        
    return chunks


class BoolToken():
    def __init__(self, value):
        self.value = value

    def isTrue(self):
        return self.value == True
        
    def setFalse(self):
        self.value = False
        
    def setTrue(self):
        self.value = True


def processesActive(processes):
    for i, processi in enumerate(processes):
        if processi.is_alive() or processi.exitcode == None:
            return True

def launchVoxelization(c, q, epoch, cancel, snpRowsList, outputFolderBasePath, ignoreCacheLimit, pdbLmdb):
    
    try:
        verbose=False

        t=time.time()


        if cancel.value:
            return

        if verbose: print("Creating rows")

        snpRows = np.array(snpRowsList, dtype="object")

        print("Combined sample: ", snpRows.shape)

        if cancel.value:
            return

        if verbose: print("Creating batches")
        allBatches = getBatches(snpRows, c)
        random.shuffle(allBatches)
        print("Created %d batches" % len(allBatches))

        nProcs = 5

        if verbose: print("Creating batch chunks")
        batchChunks = chunks(allBatches, nProcs)
        if verbose: print("--> %d" % len(batchChunks))
        print("Created %d chunks" % len(batchChunks))
        print("Done creating batch chunks (%s)" % str( time.time() - t ))

        if verbose: print("Creating inputs")

        if cancel.value:
            return


        processes = []
        for i, batchChunk in enumerate(batchChunks):

            p = Process(target=processBatches, args=(pdbLmdb, batchChunk, c, outputFolderBasePath, ignoreCacheLimit, cancel, verbose))
            p.start()
            processes.append(p)

    
    except:
        traceback.print_exc()

def cancelVoxelization(pids_sub, p):

    pid = p.pid
    p.terminate()
    p.join()
    try:
        os.system("while kill -0 %d; do sleep 1; done;" % pid)

    except:
        traceback.print_exc()
        

class DataGenerator_triple(Sequence):

    def __init__(self, model, config, prefix, trainRows, pdbLmdb):

        'Initialization'
        self.config = config
        self.epoch=1
        self.model=model
        self.voxelizationProcess = None
        self.voxelizationProcesses_subPids = None
        self.prefix = prefix
        self.removeFiles = prefix == "train"
        self.ignoreCacheLimit = prefix == "test"
        self.trainRows = trainRows
        self.pdbLmdb = pdbLmdb
        


        self.tmpFolder = os.path.join(self.config["ramDiskOutputFolderPathBase"], prefix, "%02d" % self.epoch)
        mkdir_p(self.tmpFolder)
        
        self.nFeats = getNFeats( config["nFeatsSeq"],
                                    len(config["targetAtoms"]),
                                    config["nFeatsEvo"],
                                    config["nFeatsAltRef"],
                                    config["nFeatsAllAtomDist"],
                                    config["nFeatsProtQual"],                    
                                    config["includeEvoProfs"],
                                    config["includeAlt"],
                                    config["includeAllAtomDist"],
                                    config["includeProtQual"])

        self.nVoxels_oneDim = int(config["nVoxels"][0])

        allSamples = len(self.trainRows)
        nrSplits = max(1, int( allSamples / self.config["nSamples"] ))
        rangei = np.arange(0,allSamples)
        np.random.shuffle(rangei)

        if self.config["nSamples"] > 0:
            self.idxsSplits = np.array_split(rangei, nrSplits)
        else:
            self.idxsSplits = [rangei]

        self.initVoxelization()

    def cancel(self):
        cancelVoxelization(self.voxelizationProcesses_subPids, self.voxelizationProcess)

    def initVoxelization(self):
        t = time.time()

        if self.voxelizationProcess != None:
            print("Joining process from previous epoch")
            self.cancelToken.value = True
            self.voxelizationProcess.join()
            #self.cancel()

        print("Done joining process (%s)" % str( time.time() - t ))

        self.tmpFolder = os.path.join(self.config["ramDiskOutputFolderPathBase"], self.prefix, "%02d" % self.epoch)
        print("Creating %s" % self.tmpFolder)
        mkdir_p(self.tmpFolder)

        idxsSplitIdx = (self.epoch - 1) % len(self.idxsSplits)
        idxsSplit = self.idxsSplits[idxsSplitIdx]
        self.rowsEpoch = [ self.trainRows[idx] for idx in idxsSplit ]

        print("Samples: %d batches, %d samples in current" % (len(self.idxsSplits), len(self.rowsEpoch)))

        self.queue = Queue()
        self.cancelToken = Value("b", False, lock=False)
        self.batchCounter = 0
        self.voxelizationProcess = Process(target=launchVoxelization, args=(self.config,
                                                                            self.queue,
                                                                            self.epoch,
                                                                            self.cancelToken,
                                                                            self.rowsEpoch,
                                                                            self.tmpFolder,
                                                                            self.ignoreCacheLimit,
                                                                            self.pdbLmdb))
        self.voxelizationProcess.start()

    def on_epoch_end(self):

        print("EPOCH END START", self.prefix)

        'Updates indexes after each epoch'

        if self.epoch == self.config["feature_paiHidden_startEpoch"] and self.config["feature_paiHidden"] == "True":
            print("RESETTING")
            session = K.get_session()
            for layeri in ["dense_final1", "dense_final2", "batch_final1"]:
                layer = self.model.get_layer(layeri)
                #                layer.kernel.initializer.run(session=session)
                weights_initializer = tf.variables_initializer(layer.weights)
                session.run(weights_initializer)


        self.errors = []
        self.epoch = self.epoch + 1

        if self.removeFiles:
            print("Init voxels")
            self.initVoxelization()


        print("EPOCH END END", self.prefix)

    def __len__(self):
        'Denotes the number of batches per epoch'

        if self.ignoreCacheLimit:
            nBatchesPerEpoch = 1
        else:
            nBatchesPerEpoch = int(np.floor(len(self.rowsEpoch) / self.config["batchSize"]))

        return nBatchesPerEpoch

    def loadBatchTriples(self, verbose=False):
        if verbose: print("Listing", self.prefix)

        batchFolders = os.listdir( self.tmpFolder )
        
        if verbose: print("Done Listing", self.prefix)
        while len(batchFolders) == 0:
            time.sleep(1)
            batchFolders = os.listdir( self.tmpFolder )
        
        if verbose: print("Popping", self.prefix)
        batchFolder = batchFolders.pop()
        while not os.path.exists(os.path.join(self.tmpFolder, batchFolder, ".finished")):
            print( "Waiting for ", os.path.join(self.tmpFolder, batchFolder, ".finished"), self.prefix )
            time.sleep(0.2)


        X_names = [  "tripleIdxAll", 
                     "tripleValsAll", 
                     "tripleLengthsAll", 
                     "tripleIdxGlobalAll",
                     "tripleValsGlobalAll",
                     "tripleLengthsGlobalAll"]
        
        if verbose: print("Loading", self.prefix)
        X_list = []
        for namei in X_names:
            filePath = os.path.join(self.tmpFolder, batchFolder, namei+".npy")
            npArray = np.load(filePath)
            
            X_list.append(npArray)
        
        if verbose: print("Loading 2", self.prefix)
        
        isYlist = os.path.exists(os.path.join(self.tmpFolder, batchFolder, "y_0.npy"))
        
        if isYlist:
            y= []
            i=0
            nextFilePath=os.path.join(self.tmpFolder, batchFolder, "y_%d.npy" % i)
            while os.path.exists(nextFilePath):
                y.append( np.load(nextFilePath) )
                i+=1
                nextFilePath=os.path.join(self.tmpFolder, batchFolder, "y_%d.npy" % i)

        else:
            filePath = os.path.join(self.tmpFolder, batchFolder, "y.npy")
            y = np.load(filePath)

        otherData = {}
        otherFileNames = ["sampleWeights", "multiz_geneIds", "multiz_voxelGridNNs"]
        for fileName in otherFileNames:
            filePath = os.path.join(self.tmpFolder, batchFolder, "%s.npy" % fileName)
            otherData[fileName] = np.load(filePath)

        if self.removeFiles:
            if verbose: print("Removing", os.path.join(self.tmpFolder, batchFolder))
            shutil.rmtree(os.path.join(self.tmpFolder, batchFolder))
        if verbose: print("Sending batch", self.prefix)
        return X_list, y, otherData


    def __getitem__(self, index):
        verbose=False

        X=None
        y=None

        if self.ignoreCacheLimit:
            raise Exception("NOT IMPLEMENTED; missing case when y is a list")

        else:
            X_list, y, otherData = self.loadBatchTriples(verbose=verbose)
            X = voxelizeFromTriples(X_list[0], 
                                    X_list[1], 
                                    X_list[2], 
                                    X_list[3], 
                                    X_list[4], 
                                    X_list[5], 
                                    self.nFeats, 
                                    int(self.nVoxels_oneDim))

        #print("Got index %d of %d" % (self.batchCounter, len(self)))

        self.batchCounter += 1

        if self.batchCounter == len(self) :
            print("Reached final index, deleting tmp %s" % self.tmpFolder)
            shutil.rmtree(self.tmpFolder)

        if self.config["doMultiz"]:
            X_list = [X,
                      otherData["multiz_geneIds"],
                      otherData["multiz_voxelGridNNs"]]
        else:
            X_list = X

        return X_list, y, otherData["sampleWeights"]



    


