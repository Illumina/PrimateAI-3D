import numpy as np

def getTargetAA(targetResId, targetPdbObject, converter):
    resIds = np.unique(targetPdbObject.get("resid"))
    resIdx = np.argwhere( resIds == targetResId )
    chains = np.unique(targetPdbObject.get("chain"))
    
    if resIdx.shape[0] == 0:
        return None, None
    
    resCaIdx_tmp = np.argwhere( (targetPdbObject.get("resid") == targetResId) & 
                            (targetPdbObject.get("name") == "CA") &
                            (targetPdbObject.get("chain") == chains[0] ) )
    
    atomIdxs = np.argwhere( (targetPdbObject.get("resid") == targetResId) &
                            (targetPdbObject.get("chain") == chains[0]) )
    
    atomIdxs_list = None
    resCaIdx = None
    aa = None
    
    if atomIdxs.shape[1] > 0 and resCaIdx_tmp.shape[0] > 0:
            resCaIdx = resCaIdx_tmp[0, 0]
            aa = converter.threeToOneLetterCode(targetPdbObject.get("resname")[resCaIdx])
            atomIdxs_list = atomIdxs[:,0].tolist()
    
    return tuple([aa, atomIdxs_list])


def getAaWindow(targetPdbObject, targetResId, c, windowSize):

    windowSizeLeft = int((windowSize - 1) / 2)
    windowIdxs = list(range(targetResId - windowSizeLeft, targetResId + windowSizeLeft + 1))
    
    ress = []
    for windowIdx in windowIdxs:
        res = getTargetAA(windowIdx, targetPdbObject, c)
        ress.append(res)
    
    aaWindow = [aai for aai, atomsi in ress]
    
    return aaWindow

    


def getPrevNextAAs(targetPdbObject_mol, targetResId, c):
    resIds = np.unique(targetPdbObject_mol.get("resid"))
    resIdx = np.argwhere( resIds == targetResId )[0,0]

    chains = np.unique(targetPdbObject_mol.get("chain"))

    errors = []

    prevAA=np.nan
    prevNextAtomIdxs = []
    if resIdx >= 1:
        prevResId = resIds[resIdx-1]

        prevResCaIdx_tmp = np.argwhere( (targetPdbObject_mol.get("resid") == prevResId) & 
                                    (targetPdbObject_mol.get("name") == "CA") &
                                    (targetPdbObject_mol.get("chain") == chains[0] ) )

        prevAtomIdxs = np.argwhere( (targetPdbObject_mol.get("resid") == prevResId) &
                                    (targetPdbObject_mol.get("chain") == chains[0]) )

        if prevAtomIdxs.shape[1] > 0 and prevResCaIdx_tmp.shape[0] > 0:
            prevResCaIdx = prevResCaIdx_tmp[0, 0]
            prevAA = c.threeToOneMap[targetPdbObject_mol.get("resname")[prevResCaIdx]]
            prevNextAtomIdxs.extend(prevAtomIdxs[:,0].tolist())
        else:
            errors.append("Out of range")

    nextAA=np.nan
    if resIdx + 1 < resIds.shape[0]:
        nextResId = resIds[resIdx+1]

        nextResCaIdx_tmp = np.argwhere( (targetPdbObject_mol.get("resid") == nextResId) & 
                                   (targetPdbObject_mol.get("name") == "CA") &
                                   (targetPdbObject_mol.get("chain") == chains[0] ))

        nextAtomIdxs = np.argwhere( (targetPdbObject_mol.get("resid") == nextResId) &
                                    (targetPdbObject_mol.get("chain") == chains[0]) )

        if nextAtomIdxs.shape[1] > 0 and nextResCaIdx_tmp.shape[0] > 0:
            nextResCaIdx = nextResCaIdx_tmp[0, 0]
            nextAA = c.threeToOneMap[targetPdbObject_mol.get("resname")[nextResCaIdx]]
            prevNextAtomIdxs.extend(nextAtomIdxs[:,0].tolist())
        else:
            errors.append("Out of range")


    return prevAA, nextAA, prevNextAtomIdxs, errors

