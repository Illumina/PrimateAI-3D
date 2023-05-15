import pickle
import lz4.frame
import numpy as np
from numba import njit
import pandas as pd
from multiprocessing import Pool
import globalVars

def getMulitzNumSpecies(multizLmdb):
    with multizLmdb.begin() as multizLmdbTxn:
        with multizLmdbTxn.cursor() as curs:
            for key, rec in curs:
                if "_" in key.decode("ascii"):
                    seq = pickle.loads(lz4.frame.decompress(rec))[0]
                    return seq.shape[0]

class MultizLMDB:
    def __init__(self, lmdbObj, c):
        self.lmdbObj = lmdbObj
        self.c = c
        self.generateSpeciesFilter()

    def generateSpeciesFilter(self):
        speciesDF = pd.read_csv(self.c["speciesDfFilePath"])

        self.nSpeciesTotal = self.c["multizNspecies"]

        if not "multizGapStopFeats" in self.c:
            self.c["multizGapStopFeats"] = False


        self.multizGapStopFeats = self.c["multizGapStopFeats"]

        boolMasksAliType = []
        if self.c["multizIsZoonomia"]:
            boolMasksAliType.append(speciesDF["isZoonomia"].copy())
        if self.c["multizIsZ100"]:
            boolMasksAliType.append(speciesDF["isZ100"].copy())
        if self.c["multizIsJack"]:
            boolMasksAliType.append(speciesDF["isJack"].copy())

        boolMaskAliType = pd.DataFrame(boolMasksAliType).any().values

        boolMasksPrimate = []
        if self.c["multizIsPrimate"]:
            boolMasksPrimate.append(speciesDF["isPrimate"].copy())
        if self.c["multizIsNonPrimate"]:
            boolMasksPrimate.append(speciesDF["isNonPrimate"].copy())

        boolMaskPrimate = pd.DataFrame(boolMasksPrimate).any().values

        self.boolMask = boolMaskAliType & boolMaskPrimate

        self.boolMaskLabel = self.boolMask
        if self.c["multizInvertLabels"]:
            self.boolMaskLabel = ~self.boolMaskLabel

        self.nSpecies = np.sum(self.boolMask)
        self.nSpeciesLabel = np.sum(self.boolMaskLabel)

        assert (self.nSpecies) > 0
        assert (self.nSpeciesLabel) > 0



def calcMultizLabels(snpRows, multizObj, nProcs=1):
    print("nProcs", nProcs)

    inputs = [(s, multizObj.nSpeciesTotal, multizObj.boolMaskLabel) for s in snpRows ]

    if nProcs > 1:
        p = Pool(processes=nProcs)
        newSnpRows = p.map(calcMultizLabel, inputs, chunksize = 100000)
        p.close()
        p.terminate()
        p.join()
    else:
        newSnpRows = list(map(calcMultizLabel, inputs))

    return newSnpRows


def getVarsLmdb(snpRows, lmdbObj, nSpecies):
    retArr_list = []
    with lmdbObj.begin() as lmdbTxn:
        for gene_name, change_position_1based, protPosResnamenum, labelsArr, isJigsaw in snpRows:
            r = getLmdbRec(gene_name, change_position_1based, lmdbTxn)

            if r != None:

                seqs, gapFracs, containsStop = r
                retArr_list.append( (seqs, gapFracs, containsStop, True) )
            else:
                seqs = np.full((nSpecies), 22, dtype=np.uint8)
                gapFracs = np.full((nSpecies), 0, dtype=np.uint8)
                containsStop = np.full((nSpecies), 0, dtype=np.uint8)

                retArr_list.append( (seqs, gapFracs, containsStop, False) )

    return retArr_list


def calcMultizLabel(inputi):
    snpRow, nSpeciesTotal, boolMaskLabel = inputi



    lmdbObj = globalVars.globalVars["multizLmdb"]

    gene_name, change_position_1based, protPosResnamenum, labelsArr, isJigsaw = snpRow
    multizArrs = getVarLmdb(gene_name, change_position_1based, lmdbObj, nSpeciesTotal)
    multizSeqArr = multizArrs[0]

    newLabels = updateLabel(labelsArr.astype(np.int16), multizSeqArr, boolMaskLabel)
    newLabels = newLabels.astype(np.float32)

    snpRowNew = (gene_name, change_position_1based, protPosResnamenum, newLabels, isJigsaw)

    return snpRowNew


def getVarLmdb(gene_name, change_position_1based, lmdbObj, nSpecies):

    with lmdbObj.begin() as lmdbTxn:

        r = getLmdbRec(gene_name, change_position_1based, lmdbTxn)

        if r != None:
            seqs, gapFracs, containsStop = r
            retArr =(seqs, gapFracs, containsStop, True)
        else:
            seqs = np.full((nSpecies), 22, dtype=np.uint8)
            gapFracs = np.full((nSpecies), 0, dtype=np.uint8)
            containsStop = np.full((nSpecies), 0, dtype=np.uint8)

            retArr = (seqs, gapFracs, containsStop, False)

    return retArr


def getLmdbRec(gene_name, change_position_1based, lmdbTxn):
    targetKey = "%s_%s" % (gene_name, change_position_1based)

    targetTupleObjectBytes = lmdbTxn.get(targetKey.encode("ascii"))

    if targetTupleObjectBytes == None:
        return None

    targetTupleObject = pickle.loads(lz4.frame.decompress(targetTupleObjectBytes))


    return targetTupleObject


@njit()
def updateLabel(labelArr, multizArr, boolMaskLabel):
    newLabelArr = np.full(labelArr.shape[0], -1000, dtype=np.int16)

    for i in range(labelArr.shape[0]):
        if labelArr[i] != -1000:
            newLabelArr[i] = 1

    for species_idx in range(multizArr.shape[0]):
        if boolMaskLabel[species_idx]:

            species_aa = multizArr[species_idx]

            if species_aa < 20:

                labelVal = labelArr[species_aa]

                if labelVal != -1000:
                    newLabelArr[species_aa] = 0

    return newLabelArr



def getVoxelGridNN(centerIdxToResIds, nVoxels_local):
    voxelGrid_nn = np.full((len(centerIdxToResIds), nVoxels_local * nVoxels_local * nVoxels_local, 1), -1, dtype=np.int32)

    for i in range(len(centerIdxToResIds)):
        centerIdxToResId = centerIdxToResIds[i]

        voxelGrid_nn[i, centerIdxToResId[:, 0], 0] = centerIdxToResId[:, 1]

    voxelGrid_nn = voxelGrid_nn.reshape((len(centerIdxToResIds), nVoxels_local, nVoxels_local, nVoxels_local, 1))

    return voxelGrid_nn


def getVoxelGridToNNMap(centerIdxToResIds, nvoxels, geneNameToId):

    geneIDs = []
    centerIdxToResIdLocals = []

    for gene_name, centerIdxToResIdLocal in centerIdxToResIds:

        centerIdxToResIdLocals.append(centerIdxToResIdLocal)
        geneIDs.append( geneNameToId[gene_name] )


    geneIdsArr = np.array(geneIDs)
    voxelGridNNs = getVoxelGridNN(centerIdxToResIdLocals, nvoxels)

    return geneIdsArr, voxelGridNNs
