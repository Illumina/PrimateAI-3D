
from numba import jit
import numba as nb
from numba import uint16, int32, int64, float32, typeof, types, bool_, prange
import math
import numpy as np


#Check
@jit('int32(int32,int32,int32,int32,int32,int32,bool_,bool_,bool_,bool_)', nopython=True)
def getNFeats(      nFeatsSeq,
                    nTargetAtoms,
                    nFeatsEvo,
                    nFeatsAlt,
                    nFeatsAllAtomDist,
                    nFeatsProtQual,
                    includeEvoProfs,
                    includeAlt,
                    includeAllAtomDist,
                    includeProtQual):

    nFeats = nFeatsSeq * nTargetAtoms
    
    if includeEvoProfs:
        nFeats += nFeatsEvo
    if includeAlt:
        nFeats += nFeatsAlt
    if includeProtQual:
        nFeats += nFeatsProtQual
    if includeAllAtomDist:
        nFeats += nFeatsAllAtomDist
    
    return nFeats


def getNFeats_noNumba(nFeatsSeq,
              nTargetAtoms,
              nFeatsEvo,
              nFeatsAlt,
              nFeatsAllAtomDist,
              nFeatsProtQual,
              includeEvoProfs,
              includeAlt,
              includeAllAtomDist,
              includeProtQual):
    nFeats = nFeatsSeq * nTargetAtoms
    print("Seq: %d" % (nFeatsSeq * nTargetAtoms))

    if includeEvoProfs:
        nFeats += nFeatsEvo
        print("Evo: %d" % nFeatsEvo)
    if includeAlt:
        nFeats += nFeatsAlt
        print("Alt: %d" % nFeatsAlt)
    if includeProtQual:
        nFeats += nFeatsProtQual
        print("Qual: %d" % nFeatsProtQual)
    if includeAllAtomDist:
        nFeats += nFeatsAllAtomDist
        print("Allatom: %d" % nFeatsAllAtomDist)

    return nFeats


def getTargetAACoords(boxAAs, boxAtomNames, boxCoords, boxIdxs, boxResIds, dirSignsBox, targetAA, targetAtomName):
    targetAAIdxs = boxAAs == targetAA
    
    targetAAIdxs = targetAAIdxs & (boxAtomNames==targetAtomName)
        
    targetAACoords = boxCoords[targetAAIdxs]
    targetIdxs = boxIdxs[targetAAIdxs]
    targetResIds = boxResIds[targetAAIdxs]
    targetDirSigns = dirSignsBox[targetAAIdxs]
    
    return targetAAIdxs, targetAACoords, targetResIds, targetIdxs, targetDirSigns

def getGridCenters(nVoxels, center, voxelSize):
    
    nVoxels = np.array(nVoxels)
    
    if nVoxels[0] % 2 == 0:
        raise Exception("Number of voxels must be odd!")
    
    x, y, z = nVoxels

    firstdim = np.repeat(np.arange(x) * voxelSize, y*z)

    seconddim = np.tile(np.repeat(np.arange(y) * voxelSize, z), x)

    thirddim = np.tile(np.arange(z) * voxelSize, x*y)

    combined = np.vstack((firstdim.T, seconddim.T, thirddim.T)).T.astype(np.float64)
    combined = combined.reshape([x, y, z, 3])

    nVoxelsSide = (nVoxels[0] - 1) / 2 
    minCenter = center - (nVoxelsSide * voxelSize)
    
    centers = combined + minCenter

    centers = centers.reshape(np.prod(nVoxels), 3).copy()
    
    return centers



def countAminoAcidsInBox(targetPdbObject, boxIdxs):
    
    uniqueAll, countsAll = np.unique(targetPdbObject["resid"], return_counts=True)
    countDictAll = dict(zip(uniqueAll, countsAll))

    uniqueBox, countBox = np.unique(targetPdbObject["resid"][boxIdxs], return_counts=True)
    countDictBox = dict(zip(uniqueBox, countBox))

    partial = 0
    for idi, count in countDictBox.items():
        if count != countDictAll[idi]:
            partial += 1
        
    return uniqueAll.shape[0], uniqueBox.shape[0], partial, np.unique(boxIdxs).shape[0]

#Check
@jit('float32[:](float32[:])', nopython=True)
def calcForce_angstr(dists):    
    
    for i in range(dists.shape[0]):
        dists[i] = max(0, min(1, 1.0 - (dists[i] / 5.0)))
        
    return dists

#Check
@jit('int32[:](int32, int32)', nopython=True)
def getDirSignArray(nAtoms, centralCaAtomIdx):
    indexRange = np.arange(0, nAtoms, dtype=np.int32)
    dirSignArray = np.where(indexRange < centralCaAtomIdx, -1, 1).astype(np.int32)
    return dirSignArray

@jit("float32[:,:](float32[:,:], float32[:,:])", nopython = True)
def dot_numba(A,B):
    m, n = A.shape
    p = B.shape[1]

    C = np.zeros((m,p), dtype=np.float32)

    for i in range(0,m):
        for j in range(0,p):
            for k in range(0,n):
                C[i,j] += A[i,k]*B[k,j] 
    return C

#Check
@jit("float32[:,:](float32[:,:], float32[:], float32[:,:], bool_)", nopython = True)
def centerAndRandomRotate_helper(atomCoords, centerCoord, rotMatrix, rotate):

    atomCoords_ret = np.empty(atomCoords.shape, dtype=np.float32)
    for i in range(atomCoords.shape[0]):
        atomCoords_ret[i, 0] = atomCoords[i, 0] - centerCoord[0]
        atomCoords_ret[i, 1] = atomCoords[i, 1] - centerCoord[1]
        atomCoords_ret[i, 2] = atomCoords[i, 2] - centerCoord[2]
    

    if rotate:
        atomCoords_ret = dot_numba( atomCoords_ret, rotMatrix )

    return atomCoords_ret

@jit("float32[:,:](float32[:,:], float32[:], float32[:,:,:], bool_)", nopython = True)
def centerAndRandomRotate(atomCoords, centerCoord, rotMats, rotate):

    if rotate:
        rotMat = rotMats[np.random.randint(rotMats.shape[0]),:]
    else:
        rotMat = np.array( [[1.,  0.,  0.], \
                            [0.,  1.,  0.], \
                            [0.,  0.,  1.]], dtype=np.float32)

    atomCoords_ret = centerAndRandomRotate_helper(atomCoords, centerCoord, rotMat, rotate)
    
    return atomCoords_ret

#Check
@jit('bool_[:](float32[:,:], int64[:], int64, float64, bool_)', nopython=True)
def getTargetBoxIdxs(allCoords, resIds, change_position_1based, edgeLen, excludeCentralAA):
    boxIdx = np.empty( allCoords.shape[0], dtype=np.bool_ )
    
    if excludeCentralAA:
        for i in range(allCoords.shape[0]):
            boxIdx[i] = (allCoords[i][0] < edgeLen and allCoords[i][0] > -edgeLen) and \
                        (allCoords[i][1] < edgeLen and allCoords[i][1] > -edgeLen) and \
                        (allCoords[i][2] < edgeLen and allCoords[i][2] > -edgeLen) and \
                        resIds[i] != change_position_1based
    else:
        for i in range(allCoords.shape[0]):
            boxIdx[i] = (allCoords[i][0] < edgeLen and allCoords[i][0] > -edgeLen) and \
                        (allCoords[i][1] < edgeLen and allCoords[i][1] > -edgeLen) and \
                        (allCoords[i][2] < edgeLen and allCoords[i][2] > -edgeLen)
    
    return boxIdx






########################################################
############ PREPARING BOX CENTERS #####################
########################################################


@jit('int32[:](float32[:], float32, int32)', nopython=True)
def getBoxDimIdx_helper(coord, boxLen_half, voxelSize_local):
    
    idxs = np.empty(3, dtype=np.int32)
    
    coord_std = coord[0] + boxLen_half
    idxs[0] = int(coord_std / voxelSize_local)
    
    coord_std = coord[1] + boxLen_half
    idxs[1] = int(coord_std / voxelSize_local)

    coord_std = coord[2] + boxLen_half
    idxs[2] = int(coord_std / voxelSize_local)
    
    return idxs

@jit('bool_(float32[:], float32)', nopython=True)
def withinBounds(coord, boxLen_half):
    
    if coord[0] > boxLen_half or coord[0] < -boxLen_half:
        return False
    elif coord[1] > boxLen_half or coord[1] < -boxLen_half:
        return False
    elif coord[2] > boxLen_half or coord[2] < -boxLen_half:
        return False
    
    return True
    
@jit('int32(int32[:], int32)', nopython=True)
def getCenterIdx(idxs, nVoxels_local):
    return  (idxs[2]) + \
            (idxs[1] * nVoxels_local) + \
            (idxs[0] * nVoxels_local * nVoxels_local)

@jit('int32[:](float32[:,:], float32, float32, int32)', nopython=True)
def getBoxDimIdx(coords_local, boxLen_half, voxelSize_local, nVoxels_local):
    
    centerIdxs = np.empty(coords_local.shape[0], dtype=np.int32)
    
    for i in range(coords_local.shape[0]):
        idxs_i = getBoxDimIdx_helper(coords_local[i], boxLen_half, voxelSize_local)        
        centerIdxs[i] = getCenterIdx(idxs_i, nVoxels_local)
    
    return centerIdxs


@jit('float32[:,:](float32[:], float32, float32)', nopython=True)
def getCenterNeighborCoords(singleCenterCoords, voxelSize_local, boxLen_half):
    centerNeighborCoords = np.full((3*3*3-1, 3), 1000, dtype=np.float32)

    centerNeighborCoords_loop=np.empty(3, dtype=np.float32)
    counter = 0
    for dim_j in [-1,0,1]:
        for dim_k in [-1,0,1]:
            for dim_l in [-1,0,1]:
                if not (dim_j == 0 and dim_k == 0 and dim_l == 0):

                    centerNeighborCoords_loop[0] = singleCenterCoords[0] + dim_j*voxelSize_local
                    centerNeighborCoords_loop[1] = singleCenterCoords[1] + dim_k*voxelSize_local
                    centerNeighborCoords_loop[2] = singleCenterCoords[2] + dim_l*voxelSize_local

                    isWithinBounds = withinBounds(centerNeighborCoords_loop, boxLen_half)

                    if isWithinBounds:
                        centerNeighborCoords[counter] = centerNeighborCoords_loop

                        counter += 1
                        
    return centerNeighborCoords

@jit('int32[:](float32[:,:], float32, float32, int32)', nopython=True)
def getCenterNeighborIdxs(neighborCoordArray, boxLen_half, voxelSize_local, nVoxels_local):
    
    
    
    neighborIdxs = np.full(neighborCoordArray.shape[0], -1, dtype=np.int32)
    
    for i in range(neighborCoordArray.shape[0]):
        
        currNeighborCoord = neighborCoordArray[i,:]
        
        if currNeighborCoord[0] == 1000:
            break
        
        currNeighborCoord_ext = np.ascontiguousarray(currNeighborCoord).reshape((1, -1))
        
        centerIdx = getBoxDimIdx(currNeighborCoord_ext, boxLen_half, voxelSize_local, nVoxels_local)
        neighborIdxs[i] = centerIdx[0]

    return neighborIdxs


@jit('bool_(float32[:,:], int32[:], float32[:,:])', nopython=True)
def checkNeighbors(neighborCoordArray, neighborIdxs, centerCoords):
    for i in range(neighborCoordArray.shape[0]):
        if neighborCoordArray[i,0] == 1000:
            break
        
        if not (np.all(neighborCoordArray[i,:] == centerCoords[neighborIdxs[i],:])):
            return False
    
    return True

@jit(nopython=True)
def getCenterNeighbors(centers, voxelSize_local,  boxLen_half, nVoxels_local):
    checks = []

    centerIdxToNeighborCoords = np.empty((centers.shape[0], 3*3*3-1, 3),  dtype=np.float32)
    centerIdxToNeighborIdxs = np.empty((centers.shape[0], 3*3*3-1),  dtype=np.int32)
    
    for center_i in range(centers.shape[0]):
        singleCenterCoords = centers[center_i, :]
        neighborCoordArray = getCenterNeighborCoords(singleCenterCoords, voxelSize_local, boxLen_half)
        neighborIdxs = getCenterNeighborIdxs(neighborCoordArray, boxLen_half, voxelSize_local, nVoxels_local)

        
        assert(neighborCoordArray.shape[0] == neighborIdxs.shape[0])
        
        centerIdxToNeighborCoords[center_i, :] = neighborCoordArray
        centerIdxToNeighborIdxs[center_i, :] = neighborIdxs
        
        check = checkNeighbors(centerIdxToNeighborCoords[center_i, :], centerIdxToNeighborIdxs[center_i, :], centers)
        checks.append(check)

    return centerIdxToNeighborCoords, centerIdxToNeighborIdxs, np.all(np.array(checks))
       


###############################################################
############# Calpha to Cbeta MAP #############################
###############################################################

@jit(int32[:,:](int32[:], int32[:], int32[:]), nopython=True)
def getCaCbMap(boxResIds, boxAtomNamesNum, boxAAs):

    prevResId = -1
    currCA = -1
    currCB = -1
    currAA = -1
    caCbMap = np.full((boxResIds.shape[0], 2), -1, dtype=np.int32)
    for i in range(boxResIds.shape[0]):

        currResId = boxResIds[i]
        if currResId != prevResId and prevResId != -1:
            if currCB != -1:
                caCbMap[currCA,0] = currCB

            caCbMap[currCA,1] = currAA

            currCA = -1
            currCB = -1
            currAA = -1

        currAtom = boxAtomNamesNum[i]
        if currAtom == 1:
            currCA = i
            currAA = boxAAs[i]
        elif currAtom == 2:
            currCB = i
            currAA = boxAAs[i]

        prevResId = currResId

    if prevResId != -1:
        if currCB != -1:
            caCbMap[currCA,0] = currCB
        #     print( "With CB: ", currCA )
        # else:
        #     print( "No CB: ", currCA )
        caCbMap[currCA,1] = currAA

    return caCbMap



############################################################
############### DIST #######################################
############################################################


@jit('float32[:](float32[:,:], float32[:,:])', nopython=True)
def dist3D_batch(coords_x, coords_y):
    
    res = np.empty(coords_x.shape[0], dtype=np.float32)
    
    for i in range(coords_x.shape[0]):
        val = 0
        
        tmp = coords_x[i, 0] - coords_y[i, 0]
        val += tmp * tmp
        
        tmp = coords_x[i, 1] - coords_y[i, 1]
        val += tmp * tmp
        
        tmp = coords_x[i, 2] - coords_y[i, 2]
        val += tmp * tmp
        
        res[i] = val
        
    return res

@jit('float32(float32[:], float32[:])', nopython=True)
def dist3D(coords_x, coords_y):

    val = 0

    tmp = coords_x[0] - coords_y[0]
    val += tmp * tmp

    tmp = coords_x[1] - coords_y[1]
    val += tmp * tmp

    tmp = coords_x[2] - coords_y[2]
    val += tmp * tmp

    return val

@jit('float32[:](int32[:,:], float32[:,:], float32[:,:])', nopython=True)
def calcDists(pointNeighborIdxs, coords, centers):
    maxIdx = 0
    for i in range(pointNeighborIdxs.shape[0]):
        if pointNeighborIdxs[i, 0] == -1:
            break
        maxIdx += 1

    dists = np.full((pointNeighborIdxs.shape[0]), -1, dtype=np.float32)

    for i in range(maxIdx):

        centerIdx = pointNeighborIdxs[i, 0]
        pointIdx = pointNeighborIdxs[i, 1]

        pointCoords = coords[ pointIdx ]
        centerCoords = centers[ centerIdx ]

        disti = dist3D(pointCoords, centerCoords)

        dists[i] = disti

    return dists


#############################################################
################# GETTING CENTER POINT PAIRS ################
#############################################################

@jit('int32[:](float32[:], float32, int32, int32)', nopython=True)
def getNearestBoxCenter(coord, boxLen_half, voxelSize_local, maxVoxelIdx):
    
    idxs = np.empty(3, dtype=np.int32)
    
    coord_std = coord[0] + boxLen_half
    idxs[0] = max(0, min(maxVoxelIdx, int(coord_std / voxelSize_local)))
    
    coord_std = coord[1] + boxLen_half
    idxs[1] = max(0, min(maxVoxelIdx, int(coord_std / voxelSize_local)))

    coord_std = coord[2] + boxLen_half
    idxs[2] = max(0, min(maxVoxelIdx, int(coord_std / voxelSize_local)))
    
    return idxs

@jit('int32[:,:](float32[:,:], int32[:,:], float32, float32, int32, int32)', nopython=True)
def getPointNeighborIdxs(coords, centerIdxToNeighborIdxs, boxLen_half, voxelSize_local, maxVoxelIdx, nVoxels_local):

    pointNeighborIdxs = np.full((coords.shape[0]*3*3*3, 2), -1, dtype=np.int32)

    pointToCenter = np.empty((coords.shape[0], 2), dtype=np.int32)
    for coord_i in range(coords.shape[0]):
        boxCenter = getNearestBoxCenter(coords[coord_i,:], boxLen_half, voxelSize_local, maxVoxelIdx)
        boxCenterIdx = getCenterIdx(boxCenter, nVoxels_local)
        pointToCenter[coord_i, 0] = coord_i
        pointToCenter[coord_i, 1] = boxCenterIdx

    counter = 0
    for i in range(pointToCenter.shape[0]):
        
        boxCenterIdx = pointToCenter[i, 1]
        pointIdx = pointToCenter[i, 0]
               
        neighborCenterIdxs = centerIdxToNeighborIdxs[boxCenterIdx, :]

        pointNeighborIdxs[ counter, 0 ] = boxCenterIdx
        pointNeighborIdxs[ counter, 1 ] = pointIdx

        counter += 1

        for neighbor_i in range(neighborCenterIdxs.shape[0]):
            if neighborCenterIdxs[ neighbor_i ] != -1:
                pointNeighborIdxs[ counter, 0 ] = neighborCenterIdxs[ neighbor_i ]
                pointNeighborIdxs[ counter, 1 ] = pointIdx

                #evoProfs[counter, :] = evoProf
                counter += 1


    return pointNeighborIdxs



@jit(nb.types.Tuple((int32[:,:], float32[:], float32[:,:,:], int32[:,:,:]))(int32[:,:], 
                int32[:],
                float32[:],
                int32[:],
                int32[:],
                int32,
                bool_), nopython=True)
def calcNN_jigsaw(pointNeighborIdxs, centerIdxOrder, dists, boxAtomNamesNum, boxAAs, nTargetAtoms, includePerAaDist):



    prevCenterIdx = -1
    centerCounter = 0
    for i in range(centerIdxOrder.shape[0]):
        currRowIdx = centerIdxOrder[i]
        
        currCenterIdx = pointNeighborIdxs[currRowIdx, 0]
        
        if currCenterIdx == -1: 
            continue
            
        if currCenterIdx != prevCenterIdx and prevCenterIdx != -1:
            centerCounter += 1
    
        prevCenterIdx = currCenterIdx

    if prevCenterIdx != -1:
        centerCounter += 1

    

    centerIdxToNnPointIdx = np.full((centerCounter, 2), -1, dtype=np.int32)
    centerIdxToNnDist = np.full((centerCounter), -1.0, dtype=np.float32)

    centerIdxToNnPointIdxPerAA = np.full((centerCounter, 21, nTargetAtoms), -1, dtype=np.int32)
    centerIdxToNnDistPerAA = np.full((centerCounter, 21, nTargetAtoms), 10000.0, dtype=np.float32)
    
    bestDists = np.full((21), 10000, dtype=np.float32)
    bestPointIdxs = np.full((21), -1, dtype=np.float32)
    

    prevCenterIdx = -1
    bestDist = 10000
    bestPointIdx = -1
    centerCounter = 0
    for i in range(centerIdxOrder.shape[0]):
        currRowIdx = centerIdxOrder[i]
        
        currCenterIdx = pointNeighborIdxs[currRowIdx, 0]
        
        if currCenterIdx == -1: 
            continue
        
        if prevCenterIdx != currCenterIdx and prevCenterIdx != -1:
            centerIdxToNnPointIdx[centerCounter, 0] = prevCenterIdx
            centerIdxToNnPointIdx[centerCounter, 1] = bestPointIdx
            centerIdxToNnDist[centerCounter] = bestDist
            
            bestDist = 10000.0
            bestPointIdx = -1
            bestDists[:] = 10000.0
            bestPointIdxs[:] = -1

            centerCounter += 1
            

        currPointIdx = pointNeighborIdxs[currRowIdx, 1]
        currDist = dists[currRowIdx]
    
        if currDist < bestDist:
            bestDist = currDist
            bestPointIdx = currPointIdx

        if includePerAaDist:
            currAaIdx = boxAAs[currPointIdx]
            currAtomTypeIdx = boxAtomNamesNum[currPointIdx]

            if currAtomTypeIdx != -1:

                oldDist = centerIdxToNnDistPerAA[centerCounter, currAaIdx, currAtomTypeIdx]
                    
                if currDist < oldDist:
                    centerIdxToNnDistPerAA[centerCounter, currAaIdx, currAtomTypeIdx] = currDist

                    centerIdxToNnPointIdxPerAA[centerCounter, currAaIdx, currAtomTypeIdx] =  currPointIdx

        prevCenterIdx = currCenterIdx
    
    if prevCenterIdx != -1:
        centerIdxToNnPointIdx[centerCounter, 0] = prevCenterIdx
        centerIdxToNnPointIdx[centerCounter, 1] = bestPointIdx
        centerIdxToNnDist[centerCounter] = bestDist


    return centerIdxToNnPointIdx, centerIdxToNnDist, centerIdxToNnDistPerAA, centerIdxToNnPointIdxPerAA



@jit(float32[:,:](int32[:,:], 
            float32[:,:],
            int32[:,:],
            float32[:,:],
            int32[:,:],
            float32[:,:]), nopython=True)
def getCbDists(centerIdxToNnPointIdx, centerIdxToNnDistPerAA, centerIdxToNnPointIdxPerAA, centers, caCbMap, coords):

    centerIdxToNnDistPerAA_cb = np.zeros(centerIdxToNnDistPerAA.shape, dtype=np.float32)
    
    for i in range(centerIdxToNnPointIdx.shape[0]):
        currCenterIdx = centerIdxToNnPointIdx[i,0]

        currCenterCoords = centers[currCenterIdx,:]
        
        nnPointIdxs = centerIdxToNnPointIdxPerAA[i, :]
        
        for j in range(nnPointIdxs.shape[0]):
            currNnPointIdx = nnPointIdxs[j]
            
            if currNnPointIdx != -1:
            
                cbPointIdx = caCbMap[currNnPointIdx, 0]

                if cbPointIdx == -1:
                    sqrtCaDist = np.sqrt(centerIdxToNnDistPerAA[i, j])
                    distCb = (sqrtCaDist + 1.0) * (sqrtCaDist + 1.0)
                else:
                    cbCoords = coords[cbPointIdx, :]
                    distCb = dist3D(currCenterCoords, cbCoords)

                centerIdxToNnDistPerAA_cb[i, j] = distCb
                

    return centerIdxToNnDistPerAA_cb


@jit(float32(float32, 
      int32,
      int32[:],
      float32), nopython=True)
def distToScore(dist, pointIdx, dirSignBox, maxDist):
    score = max(0, 1.0 - math.sqrt(dist / maxDist))

    if dirSignBox[pointIdx] != 1:
        score = -score
        
    return score

@jit((int32[:,:],
      float32[:],
      float32[:,:],
      int32[:,:],
      float32[:,:],
      float32[:,:],
      int32[:],
      float32[:,:],
      float32[:,:],
      int32[:],
      int32,
      int32,
      float32,
      float32[:],
      int32,
      int32[:,:]), nopython=True)
def popOcc( centerIdxToNnPointIdx, 
            centerIdxToNnDist, 
            centerIdxToNnDistPerAA, 
            centerIdxToNnPointIdxPerAA, 
            centerIdxToNnDistPerAA_cb, 
            occupancies, 
            dirSignBox, 
            protEvoArray, 
            protQualArray, 
            boxResIds, 
            nfeatsEvo, 
            nfeatsQual, 
            maxDist, 
            aaEncoded, 
            nFeatsAltRef, 
            caCbMap):
    occupancies[:, 0:nFeatsAltRef] = aaEncoded

    for i in range(centerIdxToNnPointIdx.shape[0]):
        
        centerIdx = centerIdxToNnPointIdx[i, 0]
        pointIdx = centerIdxToNnPointIdx[i, 1]

        currIndex = nFeatsAltRef

        currDist = centerIdxToNnDist[i]

        occupancies[centerIdx, currIndex] = distToScore(currDist, pointIdx, dirSignBox, maxDist)

        currIndex += 1

        resId = boxResIds[ pointIdx ]

        occupancies[centerIdx, currIndex:currIndex+nfeatsEvo] = protEvoArray[ resId, : ]

        currIndex += nfeatsEvo
        occupancies[centerIdx, currIndex:currIndex+nfeatsQual] = protQualArray[ resId, : ]

        currIndex += nfeatsQual

        nnPointIdxs = centerIdxToNnPointIdxPerAA[i, :]
        
        for j in range(nnPointIdxs.shape[0]):
            currNnPointIdx = nnPointIdxs[j]
            
            if currNnPointIdx != -1:
                
                aaIdx = j
                
                distCa = centerIdxToNnDistPerAA[i, aaIdx]
                distCb = centerIdxToNnDistPerAA_cb[i, aaIdx]
                
                occupancies[centerIdx, aaIdx+currIndex] = distToScore(distCa, currNnPointIdx, dirSignBox, maxDist)
                occupancies[centerIdx, aaIdx+currIndex+nnPointIdxs.shape[0]] = distToScore(distCb, currNnPointIdx, dirSignBox, maxDist)


@jit(int64(int32[:,:],
      float32[:],
      float32[:,:],
      int32[:,:],
      float32[:,:],
      float32[:,:],
      int32[:],
      float32[:,:],
      float32[:,:],
      int32[:],
      int32,
      int32,
      float32,
      float32[:],
      int32,
      int32[:,:]), nopython=True)
def popOcc_withAssertAndCounts( centerIdxToNnPointIdx, 
            centerIdxToNnDist, 
            centerIdxToNnDistPerAA, 
            centerIdxToNnPointIdxPerAA, 
            centerIdxToNnDistPerAA_cb, 
            occupancies, 
            dirSignBox, 
            protEvoArray, 
            protQualArray, 
            boxResIds, 
            nfeatsEvo, 
            nfeatsQual, 
            maxDist, 
            aaEncoded, 
            nFeatsAltRef, 
            caCbMap):
    occupancies[:, 0:nFeatsAltRef] = aaEncoded
    
    resIds = np.zeros((10000), dtype=np.int32)


    for i in range(centerIdxToNnPointIdx.shape[0]):
        
        centerIdx = centerIdxToNnPointIdx[i, 0]
        pointIdx = centerIdxToNnPointIdx[i, 1]

        currIndex = nFeatsAltRef

        currDist = centerIdxToNnDist[i]

        assert(centerIdx >= 0)

        occupancies[centerIdx, currIndex] = distToScore(currDist, pointIdx, dirSignBox, maxDist)

        currIndex += 1

        resId = boxResIds[ pointIdx ]

        resIds[resId] = 1

        occupancies[centerIdx, currIndex:currIndex+nfeatsEvo] = protEvoArray[ resId, : ]

        currIndex += nfeatsEvo
        occupancies[centerIdx, currIndex:currIndex+nfeatsQual] = protQualArray[ resId, : ]

        currIndex += nfeatsQual

        nnPointIdxs = centerIdxToNnPointIdxPerAA[i, :]
        
        for j in range(nnPointIdxs.shape[0]):
            currNnPointIdx = nnPointIdxs[j]
            
            if currNnPointIdx != -1:
                
                aaIdx = j
                
                distCa = centerIdxToNnDistPerAA[i, aaIdx]
                distCb = centerIdxToNnDistPerAA_cb[i, aaIdx]
                
                occupancies[centerIdx, aaIdx+currIndex] = distToScore(distCa, currNnPointIdx, dirSignBox, maxDist)
                occupancies[centerIdx, aaIdx+currIndex+nnPointIdxs.shape[0]] = distToScore(distCb, currNnPointIdx, dirSignBox, maxDist)
                

        assert(np.all(occupancies[centerIdx, :] < 2.0))
        assert(np.all(occupancies[centerIdx, :] > -2.0))

    resCount = np.sum(resIds)

    return resCount


@jit(int32(int32,
      uint16[:,:],
      float32[:],
      int32,
      int32,
      float32), nopython=True)
def addTriple(counter, returnIdxs, returnValues, centerIdx, featIdx, value):

    added = 0
    if value != 0.0:
        returnIdxs[counter, 0] = centerIdx
        returnIdxs[counter, 1] = featIdx
        returnValues[counter] = value
        
        added = 1

    return added

@jit(int32(int32,
      uint16[:,:],
      float32[:],
      int32,
      int32,
      float32[:]), nopython=True)
def addTriple_batch(counter, returnIdxs, returnValues, centerIdx, featIdx, values):
    nrAdded = 0
    for i in range(values.shape[0]):
        value = values[i]
        nrAdded += addTriple(counter+nrAdded, returnIdxs, returnValues, centerIdx, featIdx+i, value)

    return nrAdded



@jit(nb.types.Tuple((int32, uint16[:,:], float32[:]))(int32[:,:],
      float32[:],
      float32[:,:,:],
      int32[:,:,:],
      int32[:],
      float32[:,:],
      float32[:,:],
      int32[:],
      int32,
      int32,
      int32,
      int32,
      float32,
      int32,
      bool_,
      bool_), nopython=True)
def getOccTuplesWithAssert_jigsaw( centerIdxToNnPointIdx, 
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
            includeEvoProf,
            includeAaDists):
            #caCbMap):

    resIds = np.zeros((10000), dtype=np.int32)

    returnIdxs = np.full((centerIdxToNnPointIdx.shape[0] * (1 + nfeatsEvo + nfeatsQual + nfeatsSeq*nTargetAtoms), 2), -1, dtype=np.uint16)
    returnValues = np.full((centerIdxToNnPointIdx.shape[0] * (1 + nfeatsEvo + nfeatsQual + nfeatsSeq*nTargetAtoms)), -1, dtype=np.float32)

    counter = 0

    for i in range(centerIdxToNnPointIdx.shape[0]):
        
        centerIdx = centerIdxToNnPointIdx[i, 0]
        pointIdx = centerIdxToNnPointIdx[i, 1]

        currIndex = nFeatsAltRef

        currDist = centerIdxToNnDist[i]

        assert(centerIdx >= 0)

        value = distToScore(currDist, pointIdx, dirSignBox, maxDist)

        nrAdded = addTriple(counter, returnIdxs, returnValues, centerIdx, currIndex, value)

        counter += nrAdded
        currIndex += 1


        resId = boxResIds[ pointIdx ]

        resIds[resId] = 1

        if includeEvoProf:
            currEvoArray = protEvoArray[ resId, : ]

            nrAdded = addTriple_batch(counter, returnIdxs, returnValues, centerIdx, currIndex, currEvoArray)

            counter += nrAdded
            currIndex += nfeatsEvo

        currProtQualArray = protQualArray[ resId, : ]

        nrAdded = addTriple_batch(counter, returnIdxs, returnValues, centerIdx, currIndex, currProtQualArray)

        counter += nrAdded
        currIndex += nfeatsQual


        if includeAaDists:
            nnPointIdxs = centerIdxToNnPointIdxPerAA[i, :, :]
            for j in range(nnPointIdxs.shape[0]):
                for k in range(nnPointIdxs.shape[1]):
                    currNnPointIdx = nnPointIdxs[j, k]

                    if currNnPointIdx != -1:
                        
                        aaIdx = j
                        atomTypeIndex = k
                        
                        dist = centerIdxToNnDistPerAA[i, aaIdx, atomTypeIndex]

                        value = distToScore(dist, currNnPointIdx, dirSignBox, maxDist)

                        nrAdded = addTriple(counter, returnIdxs, returnValues, centerIdx, (nnPointIdxs.shape[1] * aaIdx) + atomTypeIndex + currIndex, value)

                        counter += nrAdded


    assert(np.all(returnValues[0:counter] <= 50.0))
    assert(np.all(returnValues[0:counter] >= -65.0))

    resCount = np.sum(resIds)

    return resCount, returnIdxs[0:counter,:], returnValues[0:counter]



@jit(nb.types.Tuple((uint16[:,:], float32[:]))(int32[:,:],
      float32[:],
      float32[:,:,:],
      int32[:,:,:],
      int32[:],
      float32[:,:],
      float32[:,:],
      int32[:],
      int32,
      int32,
      int32,
      int32,
      float32,
      int32), nopython=True)
def getOccTuples_jigsaw( centerIdxToNnPointIdx, 
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
            nFeatsAltRef):
            #caCbMap):


    returnIdxs = np.full((centerIdxToNnPointIdx.shape[0] * (1 + nfeatsEvo + nfeatsQual + nfeatsSeq*nTargetAtoms), 2), -1, dtype=np.uint16)
    returnValues = np.full((centerIdxToNnPointIdx.shape[0] * (1 + nfeatsEvo + nfeatsQual + nfeatsSeq*nTargetAtoms)), -1, dtype=np.float32)

    counter = 0

    for i in range(centerIdxToNnPointIdx.shape[0]):
        
        centerIdx = centerIdxToNnPointIdx[i, 0]
        pointIdx = centerIdxToNnPointIdx[i, 1]
        
        currIndex = nFeatsAltRef

        currDist = centerIdxToNnDist[i]

        value = distToScore(currDist, pointIdx, dirSignBox, maxDist)
        nrAdded = addTriple(counter, returnIdxs, returnValues, centerIdx, currIndex, value)
        counter += nrAdded
        currIndex += 1

        resId = boxResIds[ pointIdx ]

        currEvoArray = protEvoArray[ resId, : ]

        nrAdded = addTriple_batch(counter, returnIdxs, returnValues, centerIdx, currIndex, currEvoArray)
        counter += nrAdded
        currIndex += nfeatsEvo
        
        currProtQualArray = protQualArray[ resId, : ]

        nrAdded = addTriple_batch(counter, returnIdxs, returnValues, centerIdx, currIndex, currProtQualArray)
        counter += nrAdded
        currIndex += nfeatsQual

        nnPointIdxs = centerIdxToNnPointIdxPerAA[i, :, :]

        for j in range(nnPointIdxs.shape[0]):
            for k in range(nnPointIdxs.shape[1]):
                currNnPointIdx = nnPointIdxs[j, k]
                
                if currNnPointIdx != -1:
                    
                    aaIdx = j
                    atomTypeIndex = k
                    
                    dist = centerIdxToNnDistPerAA[i, aaIdx, atomTypeIndex]

                    value = distToScore(dist, currNnPointIdx, dirSignBox, maxDist)

                    nrAdded = addTriple(counter, returnIdxs, returnValues, centerIdx, (atomTypeIndex*aaIdx)+currIndex, value)
                    counter += nrAdded

    return returnIdxs[0:counter,:], returnValues[0:counter]




@jit(float32[:,:,:,:,:]( uint16[:,:],
              float32[:],
              int64[:],
              uint16[:],
              float32[:],
              int64[:],
              int32,
              int64), nopython=True, parallel=True, nogil=True)
def voxelizeFromTriples(tripleIdx, 
                        tripleVals, 
                        tripleLengthsCumsum, 
                        tripleIdxGlobal,
                        tripleValsGlobal,
                        tripleLengthsGlobalCumsum,
                        nFeats, 
                        nVoxels_local):

    nVars = tripleLengthsCumsum.shape[0] - 1

    assert( tripleLengthsCumsum.shape[0] == tripleLengthsGlobalCumsum.shape[0])
    
    nVoxels_local = int(nVoxels_local)


    occs = np.zeros( (nVars, nVoxels_local*nVoxels_local*nVoxels_local, nFeats), dtype=np.float32 )

    for i in prange(nVars):
        
        tripleIdxStart = tripleLengthsCumsum[i]
        tripleIdxEnd = tripleLengthsCumsum[i+1]
        
        for j in range(tripleIdxStart, tripleIdxEnd):
            
            currTripleIdx = tripleIdx[j]
            
            currCenterIdx = currTripleIdx[0]
            currFeatIdx = currTripleIdx[1]
            currTripleVal = tripleVals[j]

            occs[i, currCenterIdx, currFeatIdx] = currTripleVal
            

        tripleIdxGlobalStart = tripleLengthsGlobalCumsum[i]
        tripleIdxGlobalEnd = tripleLengthsGlobalCumsum[i+1]
        
        for j in range(tripleIdxGlobalStart, tripleIdxGlobalEnd):
            
            currFeatIdx = tripleIdxGlobal[j]
            currTripleVal = tripleValsGlobal[j]

            occs[i, :, currFeatIdx] = currTripleVal

    occs_reshaped = occs.reshape( nVars, nVoxels_local, nVoxels_local, nVoxels_local, nFeats )
    
    return occs_reshaped


@jit(float32[:,:,:,:,:]( uint16[:,:],
              float32[:],
              int64[:],
              uint16[:],
              float32[:],
              int64[:],
              int32,
              int64,
              int64[:]), nopython=True, parallel=True, nogil=True)
def voxelizeFromTriples_idxList(tripleIdx, 
                        tripleVals, 
                        tripleLengthsCumsum, 
                        tripleIdxGlobal,
                        tripleValsGlobal,
                        tripleLengthsGlobalCumsum,
                        nFeats, 
                        nVoxels_local,
                        varIdxs):
                        
    nVars = varIdxs.shape[0]

    assert( tripleLengthsCumsum.shape[0] == tripleLengthsGlobalCumsum.shape[0])
    
    occs = np.zeros( (nVars, nVoxels_local*nVoxels_local*nVoxels_local, nFeats), dtype=np.float32 )

    for i in prange(nVars):
        
        currVarIdx = varIdxs[i]

        tripleIdxStart = tripleLengthsCumsum[currVarIdx]
        tripleIdxEnd = tripleLengthsCumsum[currVarIdx+1]
        
        for j in range(tripleIdxStart, tripleIdxEnd):
            
            currTripleIdx = tripleIdx[j]
            
            currCenterIdx = currTripleIdx[0]
            currFeatIdx = currTripleIdx[1]
            currTripleVal = tripleVals[j]

            occs[i, currCenterIdx, currFeatIdx] = currTripleVal
            

        tripleIdxGlobalStart = tripleLengthsGlobalCumsum[currVarIdx]
        tripleIdxGlobalEnd = tripleLengthsGlobalCumsum[currVarIdx+1]
        
        for j in range(tripleIdxGlobalStart, tripleIdxGlobalEnd):
            
            currFeatIdx = tripleIdxGlobal[j]
            currTripleVal = tripleValsGlobal[j]

            occs[i, :, currFeatIdx] = currTripleVal

    occs_reshaped = occs.reshape( nVars, nVoxels_local, nVoxels_local, nVoxels_local, nFeats )
    
    return occs_reshaped

