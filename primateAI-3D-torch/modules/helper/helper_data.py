import random
import numpy as np
import torch
import collections
import pandas as pd
from helper.helper_subProt import SubProt


def getGoodAaPosIdxs(geneDict):
    isGoodDna_arr = geneDict["dna_mis"] * geneDict["dna_coveredByRefDF"]

    posIdxGood, mutIndexGood = np.where(isGoodDna_arr == 1)

    goodAaNr = geneDict["dna_altAaNum"][posIdxGood, mutIndexGood]

    changePosArr = np.arange(geneDict["dna_mis"].shape[0]).repeat(12).reshape(geneDict["dna_mis"].shape[0], -1)

    goodPosNr = changePosArr[posIdxGood, mutIndexGood]

    return goodPosNr, goodAaNr

def getJigsawLabels(geneDict, c):
    goodPosNr, goodAaNr = getGoodAaPosIdxs(geneDict)

    jigsaw_labels_all = (geneDict[c["loss_jigsaw_labelFeature"]][:, :20] > 0).astype(np.float32)

    isMisArr = np.zeros_like(jigsaw_labels_all)
    isMisArr[goodPosNr, goodAaNr] = 1.0

    jigsaw_labels = np.where(isMisArr == 1, jigsaw_labels_all, c["mask_value"])

    return jigsaw_labels


def addJigsawLabels(pdbDict, c):
    for i, geneDict in enumerate(pdbDict.values()):
        if i % 1000 == 0: print(i)

        geneDict["label_jigsaw"] = getJigsawLabels(geneDict, c)

def maskNonMisVars(targetTensor, isAccByMisTensor, maskValue):
    targetTensor_filled = targetTensor.masked_fill(isAccByMisTensor == 0, maskValue)

    return targetTensor_filled

def dnaToNum(b):
    if b == "A" or b == "a":
        return 0
    if b == "T" or b == "t":
        return 1
    if b == "G" or b == "g":
        return 2
    if b == "C" or b == "c":
        return 3


def readSnpFilePath(variantDF_filePath):
    print("Loading %s" % variantDF_filePath)

    if variantDF_filePath.endswith(".csv"):
        variantDF = pd.read_csv(variantDF_filePath, index_col=0)
    else:
        variantDF = pd.read_pickle(variantDF_filePath) #, index_col=0)

    if (not "changePos_mutPos" in variantDF) and ("nucpos_codon" in variantDF.columns): 
        variantDF["non_flipped_ref_num"] = variantDF["non_flipped_ref"].apply(dnaToNum)
        variantDF["non_flipped_alt_num"] = variantDF["non_flipped_alt"].apply(dnaToNum)
        variantDF["changePos_mutPos"] = ((variantDF["nucpos_codon"] - 1) * 4) + variantDF["non_flipped_alt_num"]
    
    if not "label_numeric_aa" in variantDF:
        variantDF['label_numeric_aa'] = variantDF['ref_aa'].replace(aaToIdx)
        variantDF['label_numeric_aa_alt'] = variantDF['alt_aa'].replace(aaToIdx)


    return variantDF


def getVariantTuples_fromFile(variantDF_filePath, mask_val, maxSamples=-1):

    variantDF = readSnpFilePath(variantDF_filePath)

    paiRows = getVariantTuples_fromDF(variantDF, mask_val, maxSamples=maxSamples)

    print("Done loading %s" % variantDF_filePath)

    return paiRows


def getVariantTuples_fromDF(variantDF, score_mask_val, maxSamples=-1, asDF=False):

    if not "label_numeric_func" in variantDF.columns:
        raise Exception("NO!")

    if not "gene_name" in variantDF.columns:
        raise Exception("Missing gene_name in DF!")

    variantDF_tmp = variantDF[["gene_name", "change_position_1based", "label_numeric_aa", "label_numeric_aa_alt", "label_numeric_func"]].dropna().copy() #
    variantDF_tmp["change_position_1based"] = variantDF_tmp["change_position_1based"].astype("int")
    variantDF_tmp["label_numeric_func"] = variantDF_tmp["label_numeric_func"].astype("int")
    variantDF_tmp["label_numeric_aa"] = variantDF_tmp["label_numeric_aa"].astype("int")
    variantDF_tmp["label_numeric_aa_alt"] = variantDF_tmp["label_numeric_aa_alt"].astype("int")

    print(variantDF_tmp)

    if maxSamples > 0:
        variantDF_tmp = variantDF_tmp.copy()
        variantDF_tmp = variantDF_tmp.sample(frac=1.0).reset_index(drop=True)
        variantDF_tmp = variantDF_tmp.head(maxSamples)

    tmpDict = collections.defaultdict(list)

    for gene_name, change_position_1based, label_numeric_aa, label_numeric_aa_alt, label_numeric_func in variantDF_tmp[["gene_name",
                                                                                                                    "change_position_1based",
                                                                                                                    "label_numeric_aa",
                                                                                                                    "label_numeric_aa_alt",
                                                                                                                    "label_numeric_func"]].values.tolist():
        tmpDict[(gene_name, change_position_1based, label_numeric_aa)].append((label_numeric_aa_alt, label_numeric_func))



    paiRows = []
    for (gene_name, change_position_1based, label_numeric_aa), varList in tmpDict.items():
        labelArr = np.full(20, score_mask_val, dtype=np.float32)

        for label_numeric_aa_alt, label_numeric_func in varList:
            # try:
            labelArr[int(label_numeric_aa_alt)] = label_numeric_func
            # except:
            #    print(name, change_position_1based, label_numeric_aa, varList)
            #    traceback.print_exc()
            #    raise Exception("ERROR!")

        paiRows.append(  (gene_name, change_position_1based, label_numeric_aa, labelArr)  )

    print("Loaded %d positions" % len(paiRows))

    if asDF:
        df = pd.DataFrame( paiRows, columns=["gene_name", "change_position_1based", "label_numeric_aa", "labelArr"] )
        df = df.sample(frac=1.0)

        return df

    else:
        paiDicts = []
        for gene_name, change_position_1based, label_numeric_aa, labelArr in paiRows:
            paiDicts.append( {"gene_name": gene_name, "change_position_1based": change_position_1based, "label_numeric_aa" :label_numeric_aa, "labelArr": labelArr} )

        random.shuffle(paiDicts)

        return paiDicts





def expandRowDF(df, scoreCols = None):
    dfRows = []

    targetCols = ["gene_name", "change_position_1based", "label_numeric_aa", "labelArr"]
    if scoreCols != None:
        for scoreCol in scoreCols:
            targetCols.append( scoreCol )

    #print("Expanding...")
    #listEleNames = targetCols[3:]
    for rowTuple in df[targetCols].values:
        rowList = rowTuple.tolist()

        listEles = rowList[3:]

        #print(listEles)

        #for scoreName, scoreList in zip(scoreCols, listEles)

        for label_numeric_aa_alt, scores in enumerate(zip(*listEles)):

            label_numeric_func = scores[0]
            if label_numeric_func > -900:

                for scoreName, scorei in zip(scoreCols, scores[1:]):

                    dfRows.append( rowList[:3] + [label_numeric_func, label_numeric_aa_alt, scoreName.replace("scores_", ""), scorei] )

    df = pd.DataFrame(dfRows, columns=targetCols[:3] + ["label_numeric_func", "label_numeric_aa_alt", "scoreName", "score"])
    #df.rename(columns={"labelArr": "label_numeric_func"}, inplace=True)

    #print(df)

    return df


def atomNamesToNum_singleProt(atomNames, targetAtoms):

    atomNamesNum = np.ascontiguousarray(np.full(atomNames.shape[0], -1, dtype=np.int16))
    for i, targetAtom in enumerate(targetAtoms):
        idxs = np.where(atomNames == targetAtom)
        atomNamesNum[idxs] = i

    #print(atomNamesNum.tolist())

    return atomNamesNum


def atomNamesToNum(pdbDict, c):

    if "atom_name" in next(iter(pdbDict.values())):
        for i, (geneName, geneDict) in enumerate(list(pdbDict.items())):
            geneDict["atom_atomNamesNum"] = atomNamesToNum_singleProt(geneDict.get('atom_name'), c["voxel_targetAtoms"])


def toPyTorchDict(pdbDict):
    for geneName, geneDict in pdbDict.items():
        torchDict = {}
        for keyi, vali in geneDict.items():
            try:
                torchDict[keyi] = torch.from_numpy(vali)
            except:
                torchDict[keyi] = vali
        pdbDict[geneName] = torchDict
    return pdbDict



def mapToProtSplitIdxs(variantDF, input_splitProtIdMappingFilePath):
    splitProtIdMappingDF = pd.read_pickle(input_splitProtIdMappingFilePath)

    variantDF_merged = variantDF.merge(splitProtIdMappingDF, on=["gene_name", "change_position_1based"])

    assert( len(variantDF_merged) == len(variantDF) )

    variantDF_merged["gene_name"] = variantDF_merged["gene_name_ext"]
    variantDF_merged["change_position_1based"] = variantDF_merged["change_position_1based_split"]

    return variantDF_merged


def mapFromProtSplitIdxs(variantDF, c):
    splitProtIdMappingDF = pd.read_pickle(c["input_splitProtIdMappingFilePath"])

    variantDF["gene_name_ext"] = variantDF["gene_name"]
    variantDF["change_position_1based_split"] = variantDF["change_position_1based"]

    variantDF = variantDF.drop(columns=["gene_name", "change_position_1based"])

    variantDF = variantDF.merge(splitProtIdMappingDF, on=["gene_name_ext", "change_position_1based_split"])

    return variantDF


def getCombinedNumberOfFeatures(pdbDict, featureNames, c, optional_pwm_layer=None):
    geneName, geneDict = next(iter(pdbDict.items()))
    dummy_changePoss = np.array([2])
    subprot = SubProt(c, geneDict, geneName, dummy_changePoss)
    if optional_pwm_layer:
        assert(len(c['input_featsDiffMultiz']) > 0)
        non_pwm_feats = []
        for feat in featureNames:
            if feat not in c['input_featsDiffMultiz']:
                non_pwm_feats.append(feat)
        cat_non_pwm_feats_0based = subprot.getCombinedFeatureTensor_0based(non_pwm_feats, "cpu")

        nFeats = (cat_non_pwm_feats_0based.shape[1] + 
                  optional_pwm_layer.n_out_features)
    else:
        assert(len(c['input_featsDiffMultiz']) == 0)
        cat_feats_0based = subprot.getCombinedFeatureTensor_0based(featureNames, "cpu")
        nFeats = cat_feats_0based.shape[1] 
    return nFeats


aa_idx_tup = [('R', 0),
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
             ('U', 20),
             (None, 21),
             ('-', 22),
             ('X', 23),
             ('pad', 24)]

aaToIdx = {aa: aaNum for aa, aaNum in aa_idx_tup}
idxToAa = [aa for aa, _ in aa_idx_tup]
