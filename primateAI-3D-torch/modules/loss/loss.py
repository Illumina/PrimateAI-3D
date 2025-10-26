import torch
import torch.nn as nn

from loss.helper_loss import mergeSubprotScoresAndLabels
from helper.helper_data import maskNonMisVars
from loss.loss_rank import getRankLossObj
from loss.loss_obs import getObsLoss, getJigsawLoss


class PaiLoss(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c

        self.rankLossObj = getRankLossObj(c)
        #TODO: move to right after config is read; determine weights based on what rank loss is selected
        c["loss_weight_rank"] = self.rankLossObj.loss_weight

        self.lossNames = [l for l in sorted(self.c["losses"]) if l != "gradnorm"]
        self.lossNameToNr = {}
        lossWeights_list = []
        for i, lossName in enumerate(self.lossNames):
            self.lossNameToNr[ lossName ] = i
            lossWeights_list.append( c["loss_weight_" + lossName] )
        self.lossWeights = torch.tensor(lossWeights_list, device=c["torch_device"])

        self.dfs = []
        self.batchCounter = 0


    def forward(self, scoreCollection):

        self.protCounter = 0
        # print("subprots:", len(scoreCollection.subprot_outputs))

        lossHistory = torch.zeros((len(scoreCollection.subprot_outputs), len(self.lossNames)), device=self.c["torch_device"])

        nLabels = {loss: 0 for loss in self.c["losses"]}
        for i, subprotOutput in enumerate(scoreCollection.subprot_outputs):

            # print("Protein", i)

            if "obs" in self.c["losses"]:
                lossName = "obs"
                lossTensor, debugDF = getObsLoss(subprotOutput["dna"],
                                                    self.gradnormObj,
                                                    self.c,
                                                    lossName)

                if self.c["loss_summary"] == "mean":
                    obsLoss = lossTensor.mean()
                elif self.c["loss_summary"] == "sum":
                    obsLoss = lossTensor.sum()

                lossHistory[i, self.lossNameToNr[lossName]] = obsLoss
                nLabels[lossName] += lossTensor.shape[0]

                #if not debugDF is None: self.dfs.append(debugDF)

            if "rank" in self.c["losses"]:
                lossName = "rank"
                protLabelDict = subprotOutput["protein"]["label"]

                #TODO also calculate with jigsaw scores?
                protein_scoreTensorFull = subprotOutput["protein"]["score"]["scores_full"].clone()

                rankloss = 0
                for score_name in self.c["loss_rank_targetLabels"]:

                    protLabel = protLabelDict[score_name].detach()
                    if self.c["loss_rank_misOnly"]:
                        protLabel = maskNonMisVars(protLabelDict[score_name], protLabelDict["prot_isAccByMis"], self.c["mask_value"])

                    #print(torch.sum(protLabel == self.c["mask_value"], axis=1))

                    rankloss_single_score = self.rankLossObj.calculate(protLabel,
                                                                       protein_scoreTensorFull)
                    rankloss += rankloss_single_score
                    nLabels["rank"] += protLabelDict[score_name].shape[0]

                if self.c["loss_summary"] == "mean":
                    rankloss_mean = rankloss.mean()
                    # need to divide by number of rank_targetLabels as we added the tensors up instead of concatenating the tensors
                    lossHistory[i, self.lossNameToNr["rank"]] = rankloss_mean / len(self.c["loss_rank_targetLabels"])
                elif self.c["loss_summary"] == "sum":
                    rankloss_sum = rankloss.sum()
                    lossHistory[i, self.lossNameToNr["rank"]] = rankloss_sum

            self.protCounter += 1

        losses_unweighted = torch.zeros(self.lossWeights.shape[0], device=self.c["torch_device"])

        for loss in ["obs", "rank"]:
            if loss in self.c["losses"]:
                if self.c["loss_summary"] == "mean":
                    lossMean = lossHistory[:, self.lossNameToNr[loss]].mean()
                if self.c["loss_summary"] == "sum":
                    lossMean = lossHistory[:, self.lossNameToNr[loss]].sum() / nLabels[loss]

                losses_unweighted[self.lossNameToNr[loss]] = lossMean

        subprotOutput_merged = mergeSubprotScoresAndLabels(scoreCollection)

        if "jigsaw" in self.c["losses"]:
            lossName = "jigsaw"
            lossTensor, debugDF = getJigsawLoss(subprotOutput_merged["protein"],
                                                  self.c,
                                                  lossName)

            losses_unweighted[self.lossNameToNr[lossName]] = lossTensor.mean()
            #if not debugDF is None: self.dfs.append(debugDF)

        if "obsAll" in self.c["losses"]:
            lossName = "obsAll"
            lossTensor, debugDF = getObsLoss(subprotOutput_merged["dna"],
                                                self.c,
                                                lossName)

            losses_unweighted[self.lossNameToNr[lossName]] = lossTensor.mean()
            #if not debugDF is None: self.dfs.append(debugDF)

        losses_weighted = losses_unweighted * self.lossWeights

        loss_weighted = losses_weighted.mean()

        lossDict = { "loss": loss_weighted }

        for j, lossName in enumerate(self.lossNames):
            lossDict["loss_unweighted_" + lossName] = losses_unweighted[ j ]
            lossDict["loss_weighted_" + lossName] = losses_weighted[ j ]

        self.batchCounter += 1

        if not self.c["loss_debug_keepDFs"]:
            self.dfs = []

        return lossDict








