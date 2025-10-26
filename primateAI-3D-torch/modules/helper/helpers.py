import os
import pickle
import subprocess
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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
        bboneNames = ["C", "CA", "N", "O"]
        targetPdbDict["bbone"] = np.where(np.isin(targetPdbDict["name"], bboneNames), True, False)
        bboneFilter = np.argwhere(targetPdbDict["bbone"])
        for keyi in targetPdbDict.keys():
            targetPdbDict[keyi] = targetPdbDict[keyi][bboneFilter].squeeze()
    elif atomsConfig == "calpha":
        bboneNames = ["CA"]
        targetPdbDict["bbone"] = np.where(np.isin(targetPdbDict["name"], bboneNames), True, False)
        bboneFilter = np.argwhere(targetPdbDict["bbone"])
        for keyi in targetPdbDict.keys():
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


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def chunks(list_a, chunk_size):
    r = []
    for i in range(0, len(list_a), chunk_size):
        r.append(list_a[i:i + chunk_size])

    return r

def get_git_revision_hash():
    """In case code is run in .git, extract the commit"""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except:
        return ""

def get_git_revision_short_hash():
    """In case code is run in .git, extract the commit"""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except:
        return ""
