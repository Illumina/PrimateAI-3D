import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

labelTuples = [('R', 0),
             ('I', 1),
             ('D', 2),
             ('L', 3),
             ('T', 4),
             ('P', 5),
             ('S', 6),
             ('E', 7),
             ('V', 8),
             ('H', 9),
             ('G', 10),
             ('Q', 11),
             ('N', 12),
             ('C', 13),
             ('A', 14),
             ('K', 15),
             ('M', 16),
             ('Y', 17),
             ('F', 18),
             ('W', 19),
             ('-', 20),
             (None, 21),
             ('X', 22),
             ("U", 23)]

labelDict = {aa: aaNum for aa, aaNum in labelTuples}
labelDictRev = {aaNum: aa for aa, aaNum in labelTuples}
labelAaList = [aa for aa, aaNum in labelTuples ]


def loadPdbFile(targetUpACC, targetPdbFileRel, spBase):
    targetFilePath = os.path.join(spBase, targetUpACC[:2], targetUpACC[2:4], targetUpACC[4:].split("-")[0], "swissmodel", targetPdbFileRel+".mkit.pkl")
    targetPdbObject = pickle.load(open(targetFilePath, "rb"))
    return targetPdbObject
    

def getCentralCaAtomCoords(targetResId, targetPdbObject, randomProtPos):
    
    if randomProtPos:
        resids = np.unique(targetPdbObject.get("resid"))
        targetResId = np.random.choice(resids)
    
    centralCaAtomIdx = np.argwhere( (targetPdbObject.get("resid") == targetResId) & (targetPdbObject.get("name") == "CA") )[0,0]

    centralCaAtomCoords = targetPdbObject.get("coords")[centralCaAtomIdx]

    pdbAA = targetPdbObject.get("resname")[centralCaAtomIdx]
    
    return centralCaAtomCoords, pdbAA, centralCaAtomIdx

   
def reduceAtoms(targetPdbDict, atomsConfig):
    if atomsConfig == "all":
        pass
    elif atomsConfig == "bbone":
        targetKeys = [keyi for keyi in targetPdbDict.keys() if (keyi != "evoArray" and keyi != "annoArray" and keyi != "paiArray")]
        bboneNames = ["C", "CA", "N", "O"]
        targetPdbDict["bbone"] = np.where(np.isin(targetPdbDict["name"], bboneNames), True, False)
        bboneFilter = np.argwhere(targetPdbDict["bbone"])
        for keyi in targetKeys:
            targetPdbDict[keyi] = targetPdbDict[keyi][bboneFilter].squeeze()
    elif atomsConfig == "calpha":
        targetKeys = [keyi for keyi in targetPdbDict.keys() if (keyi != "evoArray" and keyi != "annoArray" and keyi != "paiArray")]
        bboneNames = ["CA"]
        targetPdbDict["bbone"] = np.where(np.isin(targetPdbDict["name"], bboneNames), True, False)
        bboneFilter = np.argwhere(targetPdbDict["bbone"])
        for keyi in targetKeys:
            targetPdbDict[keyi] = targetPdbDict[keyi][bboneFilter].squeeze()
    else:
        raise Exception("No atoms")

    return targetPdbDict
    

def checkAaConsistency(pdbAA, upAA, c):
    try:
        pdbAA = c.threeToOneMap[pdbAA]
    except:
        pdbAA = "UNK"

    if (pdbAA != upAA) and (pdbAA != "UNK"):
        return["Error!!!!"]
    else:
        return []

def aaCharToNum(aa):
    if not aa in labelDict:
        aa = "U"

    return( labelDict[aa] )


def aaNumToChar(aaNum):

    if not aaNum in labelDictRev:
        aa = "U"
    else:
        aa = labelDictRev[aaNum]

    return( aa )

def getlabelAaList():
    return labelAaList
        

def getBvals(molBvalsRaw):
    bvalsRawFilt = molBvalsRaw[molBvalsRaw != 99.0]
    bvalsRawFilt_std = np.std(bvalsRawFilt)

    if bvalsRawFilt_std == 0:
        bvalsNorm = np.repeat(99.0, molBvalsRaw.shape[0]) #.astype("float32")
    else:
        bvalsNorm = (molBvalsRaw - np.mean(bvalsRawFilt)) / bvalsRawFilt_std

    if np.any(np.isnan(bvalsNorm)):
        bvalsNorm = np.repeat(99.0, bvalsNorm.shape[0]) #.astype("float32")

    badBvalIndex = (molBvalsRaw == 99.0) | (bvalsNorm == 99.0)

    molBvalsRaw[badBvalIndex] = 0
    bvalsNorm[badBvalIndex] = 0
    
    bvalsNorm = MinMaxScaler().fit_transform(bvalsNorm[:, np.newaxis]).ravel()
    
    return molBvalsRaw, bvalsNorm


