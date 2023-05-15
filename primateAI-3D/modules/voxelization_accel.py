import pickle

from numba import jit
import numpy as np

import numba as nb
from numba import uint16, int32, int64, float32, typeof, types, boolean

from globalVars import SAMPLE_JIGSAW
from voxelization_accel_helpers import getDirSignArray, centerAndRandomRotate, getTargetBoxIdxs, calcNN_jigsaw
from voxelization_accel_helpers import getPointNeighborIdxs, calcDists, getOccTuplesWithAssert_jigsaw
import globalVars



def calcOccupancy_triples(  coords, 
                    centerIdxToNeighborIdxs, 
                    boxLen_half, 
                    voxelSize_local, 
                    maxVoxelIdx, 
                    nVoxels_local, 
                    centers, 
                    protEvoArray, 
                    protQualArray, 
                    boxResIds, 
                    dirSignBox, 
                    nfeatsEvo, 
                    nfeatsQual, 
                    nfeatsSeq,
                    nTargetAtoms,
                    maxDist, 
                    nFeatsAltRef, 
                    boxAtomNamesNum,
                    boxAAs,
                    voxelizeWithAsserts,
                    includeEvoProfs,
                    includePerAaDists):
    pointNeighborIdxs = getPointNeighborIdxs(coords, centerIdxToNeighborIdxs, boxLen_half, voxelSize_local, maxVoxelIdx, nVoxels_local)

    dists = calcDists(pointNeighborIdxs, coords, centers)
    
    centersSortedIdxs = np.argsort(pointNeighborIdxs[:,0]).astype(np.int32)
    
    centerIdxToNnPointIdx, centerIdxToNnDist, centerIdxToNnDistPerAA, centerIdxToNnPointIdxPerAA = calcNN_jigsaw(pointNeighborIdxs, centersSortedIdxs, dists, boxAtomNamesNum, boxAAs, nTargetAtoms, includePerAaDists)

    if voxelizeWithAsserts:
        centerIdxToResIds = np.empty(centerIdxToNnPointIdx.shape, dtype=np.uint16)
        centerIdxToResIds[:,0] = centerIdxToNnPointIdx[:, 0]
        centerIdxToResIds[:,1] = boxResIds[centerIdxToNnPointIdx[:, 1]]

        resIdCount, tripleIdxs, tripleVals =  getOccTuplesWithAssert_jigsaw(    centerIdxToNnPointIdx, 
                                                                                centerIdxToNnDist, 
                                                                                centerIdxToNnDistPerAA, 
                                                                                centerIdxToNnPointIdxPerAA,
                                                                                dirSignBox, 
                                                                                protEvoArray, 
                                                                                protQualArray, 
                                                                                boxResIds, 
                                                                                nfeatsEvo, 
                                                                                nfeatsQual,
                                                                                nfeatsSeq,
                                                                                nTargetAtoms,
                                                                                maxDist, 
                                                                                nFeatsAltRef,
                                                                                includeEvoProfs,
                                                                                includePerAaDists)



    else:
        raise Exception("Not implemented")

    return resIdCount, tripleIdxs, tripleVals, centerIdxToResIds
    





@jit(nb.types.Tuple((uint16[:], float32[:]))(float32[:]), nopython=True)
def getAaEncTriples(aaEnc):
    idxs = np.zeros((2), dtype=np.uint16)
    vals = np.zeros((2), dtype=np.float32)

    counter = 0
    for i in range(aaEnc.shape[0]):
        if aaEnc[i] == 1:
            idxs[counter] = i
            vals[counter] = 1

            counter += 1

    assert(counter == 2)

    return idxs, vals


def calcMaxDist(voxelSize):
    voxelDiameter = voxelSize * 3.4641 #2 * math.sqrt(3)
    return voxelDiameter

def voxelize_triples(   pdbTxn,
                targetSnpDF_local_oneRow, 
                centers, 
                edgeLen, 
                c,
                centerIdxToNeighborCoords,
                centerIdxToNeighborIdxs,
                maxVoxelIdx,
                boxLen_half,
                voxelSize_local,
                nVoxels_local):

    gene_name = targetSnpDF_local_oneRow[0]
    change_position_1based = targetSnpDF_local_oneRow[1]

    isJigsaw = targetSnpDF_local_oneRow[4] == SAMPLE_JIGSAW

    targetPdbObjectBytes = pdbTxn.get(gene_name.encode("ascii"))
    targetPdbObject = pickle.loads(targetPdbObjectBytes)

    centralCaAtomCoords = targetPdbObject.get('caArray')[change_position_1based]
    centralCaAtomIdx = targetPdbObject.get('caIndexArray')[change_position_1based][0]

    assert(targetPdbObject["resid"][int(centralCaAtomIdx)] == change_position_1based)

    dirSignArray = getDirSignArray(targetPdbObject.get('element').shape[0], centralCaAtomIdx)

    allCoords = targetPdbObject.get('coords')
    allCoords = centerAndRandomRotate(allCoords, centralCaAtomCoords, globalVars.globalVars["rotMatrices"], c["rotate"])

    if isJigsaw:
        boxIdx = getTargetBoxIdxs( allCoords, targetPdbObject["resid"], change_position_1based, edgeLen, c["excludeCentralAA"] )
    else:
        boxIdx = getTargetBoxIdxs( allCoords, targetPdbObject["resid"], change_position_1based, edgeLen, False )

    boxCoords = allCoords[boxIdx]
    boxResIds = targetPdbObject["resid"][boxIdx].astype(np.int32)
    boxAtomNames = targetPdbObject['name'][boxIdx]
    boxAAs = targetPdbObject["resnamenum"][boxIdx].astype(np.int32)
    protEvoArray = np.ascontiguousarray(targetPdbObject["feat_cons"])
    protQualArray = np.ascontiguousarray(targetPdbObject["qualArray"])

    if "doPai" in c and ((not c["doPai"]) or (not c["doJigsaw"])):
        protEvoArray[:,21] = 0
        protEvoArray[:,22] = 1
    elif isJigsaw:
        protEvoArray[:,21] = 1
        protEvoArray[:,22] = 0
    else:
        protEvoArray[:,21] = 0
        protEvoArray[:,22] = 1

    if not c["excludeCentralAA"] and isJigsaw:
        raise Exception("not implemented")

    dirSignsBox = dirSignArray[boxIdx]

    targetAtoms = c["targetAtoms"]
    boxAtomNamesNum =  np.ascontiguousarray(np.full(boxAtomNames.shape[0], -1,  dtype=np.int32))
    for i, targetAtom in enumerate(targetAtoms):
        idxs = np.where(boxAtomNames == targetAtom)
        boxAtomNamesNum[idxs] = i


    resIdCount, tripleIdxs, tripleVals, centerIdxToResIds = calcOccupancy_triples(boxCoords,
                    centerIdxToNeighborIdxs, 
                    boxLen_half, 
                    voxelSize_local, 
                    maxVoxelIdx,
                    nVoxels_local, 
                    centers, 
                    protEvoArray, 
                    protQualArray, 
                    boxResIds, 
                    dirSignsBox, 
                    c["nFeatsEvo"], 
                    c["nFeatsProtQual"],
                    c["nFeatsSeq"],
                    len(c["targetAtoms"]),
                    c["distanceUpperBound"] * c["distanceUpperBound"], 
                    c["nFeatsAltRef"], 
                    boxAtomNamesNum,
                    boxAAs,
                    c["voxelizeWithAsserts"],
                    c["includeEvoProfs"],
                    c["includePerAaDists"])

    idxsGlobals = np.zeros((2), dtype=np.uint16)
    valsGlobal = np.zeros((2), dtype=np.float32)


    countsTuple = None
    if c["voxelizeWithAsserts"]:
        countsTuple = (np.unique(boxResIds).shape[0], resIdCount)

    return tripleIdxs.astype(np.uint16), tripleVals, idxsGlobals, valsGlobal, countsTuple, centerIdxToResIds




    
