import torch
from loss.loss_rank import masked_mean

def combineMaskedScoreTensor(geneDict, targetLabels, mask_value=-1000, scoreCombMethod='mean', commonVariants=False):
    """Combine scores taking into account masked values"""
    concatenated = []
    # go over all externalScores and combine them along a 3rd dimension into 
    for externalScore in targetLabels:
        if "combinedScore" != externalScore:
            concatenated.append(geneDict[externalScore][:, :, None])
    concatenatedTensor = torch.cat(concatenated, dim=2)
    concatMask = concatenatedTensor != mask_value

    # COMBINE SCORES
    if scoreCombMethod == "mean":
        combinedScore = masked_mean(concatenatedTensor, concatMask, dim=2, keepdim=False)
    elif scoreCombMethod == "max":
        # set the mask to -infinity to ensure its not selected in a maximum calculation
        concatenatedTensor = torch.where(concatMask, concatenatedTensor, float('-inf'))
        combinedScore, _ = torch.max(concatenatedTensor, dim=2, keepdim=False)
    elif scoreCombMethod == "min":
        # set the mask to + infinity to ensure its not selected in a maximum calculation
        concatenatedTensor = torch.where(concatMask, concatenatedTensor, float('inf'))
        combinedScore, _ = torch.min(concatenatedTensor, dim=2, keepdim=False)
    else:
        raise ValueError(f"scoreCombMethod not implemented {scoreCombMethod}")
    
    # UPDATED MASK ACROSS SCORES
    if commonVariants:
        # all scores have to have a value of 1 at the location for the score to be included
        mask_sum = torch.sum(concatMask, dim=2)
        # subtracting -1 for "combinedScore"
        aggregated_mask = torch.where(mask_sum==len(targetLabels)-1, 1, 0)
    else:
        # calculate a mask union, i.e. an element has a value of 1 if at least one of the scores had one
        aggregated_mask, _ = torch.max(concatMask, dim=2)

    # RE-INTRODUCE the mask as above operations modified it
    combinedScore = torch.where(aggregated_mask.bool(), combinedScore, mask_value)
    return combinedScore

def combineScores(pdbDict, c):
    for gene_name, geneDict in pdbDict.items():
        geneDict["combinedScore"] = combineMaskedScoreTensor(geneDict, c["loss_rank_targetLabels"]+[s.replace("/", "_") for s in c["loss_rank_externalScores"]], 
                                                        mask_value=c['mask_value'], 
                                                        scoreCombMethod=c["loss_rank_scoreCombMethod"], 
                                                        commonVariants=c["loss_rank_commonVariants"])

        