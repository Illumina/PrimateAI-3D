import pandas as pd
import numpy as np
import torch
import helper.globalVars as globalVars

def addSampleWeights(pdbDict, c):
    obsTrinucArrs = []

    print("Gathering obs")

    obsType = c["dna_obsType"]
    obsTypeQual = c["dna_obsTypeQual"]
    obsTypeQualThresh = c["dna_obsTypeQualThresh"]

    print("Obs sample config", obsType, obsTypeQual, obsTypeQualThresh)

    nDropped = 0
    for i, (geneName, geneDict) in enumerate(list(pdbDict.items())):


        trinuxArr = geneDict["dna_trinuc"] #the trinuc of the variant, encoding as integer running from 0-193
        obsArr = geneDict[obsType] #whether the variant has been observed
        isCov = geneDict["dna_coveredByRefDF"] == 1 #whether the variant comes from a protein position that can or should have labels
        isMis = geneDict["dna_mis"] == 1 #whether the variant is a missense variant

        qualArr = geneDict[obsTypeQual] # the quality field of the observation, i.e. how confident are we that the variant has (not) been observed

        qualPassingArr = (qualArr >= obsTypeQualThresh)

        if c["dna_obsSwitch"]:
            obsArr = np.where(~qualPassingArr, 0, obsArr)

        isGoodTrinuc = geneDict["dna_trinuc"] > -1

        #print(geneName, np.where(isCov))

        isHq = (((obsArr == 1) & qualPassingArr) | (obsArr == 0)) & isGoodTrinuc & isMis & isCov

        obsArr_hq = np.where(isHq, obsArr, -1)

        # geneDict[obsType + "_isHq"] = isHq
        geneDict[obsType + "_hq"] = obsArr_hq


        if not geneName in globalVars.tooLongGenes:

            obsTrinucArr = np.stack([obsArr_hq.flatten(),
                                     trinuxArr.flatten()]).transpose()

            obsTrinucArrs.append(obsTrinucArr)
        else:
            nDropped += 1

    print("Excluded %d genes" % nDropped)

    print("Counting obs")
    obsTrinucArr = np.concatenate(obsTrinucArrs, axis=0)
    obsTrinucTensor = torch.from_numpy(obsTrinucArr).to(device=c["torch_device"])
    if c["torch_device"].type == "cpu":
        print("... with numpy")
        obsTrinuc, counts = np.unique(obsTrinucTensor, axis=0, return_counts=True)
    else:
        obsTrinuc, counts = torch.unique(obsTrinucTensor, dim=0, return_counts=True)
        obsTrinuc = obsTrinuc.cpu().numpy()
        counts = counts.cpu().numpy()

    print("Calculating weights")
    obsDF = pd.DataFrame({"obs": obsTrinuc[:, 0], "trinuc": obsTrinuc[:, 1], "count": counts})

    grpdDF = obsDF.set_index(["trinuc", 'obs'])['count'].unstack(fill_value=0).reset_index()

    grpdDF.columns = ["trinuc_idx", "na", "noCount", "yesCount"]

    grpdDF = grpdDF[grpdDF["trinuc_idx"] != -1].copy()

    grpdDF["noMult"] = grpdDF["yesCount"] / grpdDF["noCount"]
    grpdDF["yesMult"] = 1.0
    grpdDF["check"] = grpdDF["noCount"] * grpdDF["noMult"]
    grpdDF["total"] = grpdDF["noCount"] + grpdDF["yesCount"]
    grpdDF["totalMult"] = grpdDF["total"] / (grpdDF["yesCount"] * 2)
    grpdDF["check2"] = (grpdDF["yesCount"] * grpdDF["totalMult"]) + (grpdDF["noCount"] * grpdDF["noMult"] * grpdDF["totalMult"])
    grpdDF["noMult_final"] = grpdDF["totalMult"] * grpdDF["noMult"]
    grpdDF["yesMult_final"] = grpdDF["totalMult"] * grpdDF["yesMult"]

    grpdDF["multRatio"] = grpdDF["yesMult_final"] / grpdDF["noMult_final"]
    grpdDF["multRatio"] = np.power(grpdDF["multRatio"], c["dna_sampleLossWeightsExp"])

    grpdDF["noMult_final"] = grpdDF["yesMult_final"] / grpdDF["multRatio"]
    grpdDF["yesMult_final"] = (grpdDF["total"] - (grpdDF["noCount"] * grpdDF["noMult_final"])) / grpdDF["yesCount"]

    sampleWeightArr_tmp = grpdDF.sort_values("trinuc_idx")[["noMult_final", "yesMult_final"]].values

    sampleWeightArr = np.zeros((sampleWeightArr_tmp.shape[0] + 1, sampleWeightArr_tmp.shape[1]), dtype=np.float32)

    sampleWeightArr[1:, :] = sampleWeightArr_tmp

    # fieldName = "dna_lossWeight_%s" % c["dna_obsType"]

    print("Adding weights to PDB dict")
    for i, (geneName, geneDict) in enumerate(pdbDict.items()):
        if i % 1000 == 0: print(i)

        # trinuxArr = geneDict["dna_trinuc"]
        # obsArr = geneDict[obsType]
        # covArr = geneDict["dna_coveredByRefDF"]

        trinucIdxArr = geneDict["dna_trinuc"].reshape(-1).astype(np.int64)

        # obsArr = geneDict[obsType].reshape(-1).astype(np.int64)
        obsHqArr = geneDict[obsType + "_hq"].reshape(-1).astype(np.int64)
        # qualPassingArr = geneDict[obsType + "_isHq"]
        isValidArr = geneDict[obsType + "_hq"] > -1

        obsHqArrIdx = np.maximum(obsHqArr, 0)

        geneDict["dna_lossWeight"] = sampleWeightArr[trinucIdxArr, obsHqArrIdx].reshape(geneDict["dna_humanObs"].shape[0], -1)
        geneDict["dna_lossWeight"] = geneDict["dna_lossWeight"] * isValidArr + np.where(isValidArr, 0, -1)

        # del geneDict[obsType+"_isHq"]


