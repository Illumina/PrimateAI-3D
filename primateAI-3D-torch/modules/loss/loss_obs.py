import pandas as pd
import torch.nn as nn
import torch


def getJigsawLoss(subprotOutput_merged_protein,
                  c,
                  lossName):

    loss_fn = getLossFn(c)

    scoreTensor_all = subprotOutput_merged_protein["score"]["scores_jigsaw"].clone()
    labelTensor_all = subprotOutput_merged_protein["label"]["label_jigsaw"]
    lossTensor = loss_fn(scoreTensor_all, labelTensor_all) * (labelTensor_all != torch.tensor(c["mask_value"]))

    debugDF = None

    return lossTensor, debugDF


def getObsLoss(subprotOutput_merged_dna,
                  c,
                  lossName):

    loss_fn = getLossFn(c)

    scoreTensor_all = subprotOutput_merged_dna["score"]["scores_full"].clone()
    labelTensor_all = subprotOutput_merged_dna["label"]["obs"]
    labelWeight_all = subprotOutput_merged_dna["label"]["obs_sampleWeight"]

    lossTensor = loss_fn(scoreTensor_all, labelTensor_all)

    lossTensor_weighted = lossTensor
    if c["loss_sampleWeights"]:
        lossTensor_weighted = lossTensor_weighted * labelWeight_all  #

    debugDF = None
    # if c["eval_keepScoresDF"] != "no":
    #
    #     colDict = dict([("loss", lossTensor[:, 0].detach().cpu().numpy()),
    #                           ("lossWeighted", lossTensor_weighted[:, 0].detach().cpu().numpy()),
    #                           ("score", scoreTensor_all[:, 0].detach().cpu().numpy()),
    #                           ("label", labelTensor_all[:, 0].detach().cpu().numpy()),
    #                           ("labelWeight", labelWeight_all[:, 0].detach().cpu().numpy()),
    #                           ("trinuc_idx", subprotOutput_merged_dna["label"]["trinuc"].detach().cpu().numpy()),
    #                           ("aaNumAlt", subprotOutput_merged_dna["label"]["aaNumAlt"].detach().cpu().numpy()),
    #                           ("aaNumRef", subprotOutput_merged_dna["label"]["aaNumRef"].detach().cpu().numpy()),
    #                           ("batchPos0based", subprotOutput_merged_dna["label"]["batchPos0based"].detach().cpu().numpy()),
    #                           ("obs_sampleWeight", subprotOutput_merged_dna["label"]["obs_sampleWeight"][:, 0].detach().cpu().numpy()),
    #                           ("lossName", lossName),
    #                           ("requires_grad", scoreTensor_all.requires_grad)])
    #
    #
    #     debugDF = pd.DataFrame(colDict)

    return lossTensor_weighted, debugDF


_loss_fn = None
def getLossFn(c):
    global _loss_fn

    if not _loss_fn is None:
        return _loss_fn
    else:
        if c['loss_obs_type'] == 'CE':
            _loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(c['torch_device'])
        elif c['loss_obs_type'] == 'MSE':
            _loss_fn = nn.MSELoss(reduction="none").to(c["torch_device"])

        return _loss_fn
