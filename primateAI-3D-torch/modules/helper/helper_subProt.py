
import torch
import numpy as np
import pandas as pd


class SubProt():
    def __init__(self, c, targetPdbObject, gene_name, changePoss):
        self.c = c
        self.targetPdbObject = targetPdbObject
        self.gene_name = gene_name

        self.changePoss = torch.tensor(changePoss) #torch.tensor(changePoss, device=c["torch_device"])

        self.outputDict = {"dna": {"label": {}, "score": {}},
                           "protein": {"label": {}, "score": {}}}

        self.multiz_extra = self.c["multiz_featGaps"] or self.c["multiz_featStop"] or self.c["multiz_featGlobal"]

        self.centralCaAtomCoords = self.targetPdbObject['caCoords'][self.changePoss, :]


    def toDevice_labels(self):
        changePoss = self.changePoss.to(device=self.c["torch_device"])
        labelDict_prot = self.getPossLabels_protein_torch(self.targetPdbObject, changePoss)
        labelDict_dna = self.getDnaLabelDict_torch(self.targetPdbObject, changePoss)

        self.outputDict["dna"]["label"] = labelDict_dna
        self.outputDict["protein"]["label"] = labelDict_prot


    def getCombinedFeatureTensor_0based(self, featureList, device):
        featureTensors = []

        for feature in featureList:
            tensor_local = self.targetPdbObject[feature]
            featureTensors.append(tensor_local)

        featAllTensor = torch.concat(featureTensors, dim=1).to(device=device).to(torch.float32)
        featAllTensor_0based = featAllTensor[1:, :]

        return featAllTensor_0based

    def getDictTensor_0based(self, featureList, device):
        featureTensors = {}
        for feature in featureList:
            tensor_local = self.targetPdbObject[feature][1:].to(device)
            featureTensors[feature] = tensor_local
        return featureTensors

    def getPossLabels_protein_torch(self, prot, posArr):
        labelDict_prot={}

        for labelName in self.c["targetLabels"]:
            #try:
            labelDict_prot[labelName] = prot[labelName].to(device=self.c["torch_device"])[posArr,:]
            # except:
            #     print(self.gene_name, labelName)
            #     print( prot[labelName].shape )
            #     print( posArr.cpu().numpy().tolist() )
            #     raise Exception("ERROR")

        return labelDict_prot


    def getBatchLabelFeatDict_torch(self, targetPdbObject, changePossGPU, obsField, trinucFeat):

        featArrDict = {}
        for feat in [obsField, "dna_mis", "dna_lossWeight", "dna_altAaNum", trinucFeat, "feat_refonehot"]:
            #print(targetPdbObject, feat, changePossGPU)
            featTensor_batch = targetPdbObject[feat].to(device=self.c["torch_device"])[changePossGPU, :]
            featArrDict[feat] = featTensor_batch

        return featArrDict


    def getDnaLabelDict_torch(self, targetPdbObject, changePossGPU):
        obsField = self.c["dna_obsType"] + "_hq"
        trinucFeat = "dna_trinuc"
        featArrDict = self.getBatchLabelFeatDict_torch(targetPdbObject, changePossGPU, obsField, trinucFeat)

        batchChangePos_0based, mutPos_0based = torch.where((featArrDict["dna_mis"] == 1) & (featArrDict[obsField] > -1))

        dna_lossWeight = featArrDict["dna_lossWeight"][batchChangePos_0based, mutPos_0based]
        dna_altAaNum = featArrDict["dna_altAaNum"][batchChangePos_0based, mutPos_0based].to(torch.int64)
        dna_obsMis = featArrDict[obsField][batchChangePos_0based, mutPos_0based]
        dna_trinuc = featArrDict[trinucFeat][batchChangePos_0based, mutPos_0based].to(torch.int64)

        dna_altRefNum = torch.where(featArrDict["feat_refonehot"])[1]
        assert dna_altRefNum.shape[0] == featArrDict["feat_refonehot"].shape[0]
        dna_altRefNum = dna_altRefNum[batchChangePos_0based]

        labelDict_dna = {"obs": dna_obsMis.unsqueeze(1).to(torch.float32),
                         "obs_sampleWeight": dna_lossWeight.unsqueeze(1).to(torch.float32),
                         "trinuc": dna_trinuc,
                         "batchPos0based": batchChangePos_0based,
                         "aaNumRef": dna_altRefNum,
                         "aaNumAlt": dna_altAaNum}

        return labelDict_dna


    def toDF(self):

        atom_coords = self.atom_coords.cpu()
        atom_atomNamesNum = self.atom_atomNamesNum.cpu()
        atom_resid = self.atom_resid.cpu()
        atom_resnamenum = self.atom_resnamenum.cpu()

        df = pd.DataFrame({"atom_coords_x": atom_coords[:,0],
                           "atom_coords_y": atom_coords[:,1],
                           "atom_coords_z": atom_coords[:,2],
                           "atom_atomNamesNum": atom_atomNamesNum,
                           "atom_resid": atom_resid,
                           "atom_resnamenum": atom_resnamenum})

        df_prot = pd.DataFrame({   "resId": range(0, self.caCoords.shape[0]),
                                   }) #"feats": self.prot_feats.tolist()

        df = df.merge(df_prot, left_on=["atom_resid"], right_on=["resId"])

        return df



def getDnaLabelDict(targetPdbObject, changePoss, c ):
    obsField = c["dna_obsType"] + "_hq"

    dna_batch_changePosMutPos, \
    dna_batch_lossWeight, \
    dna_batch_altAaNum, \
    dna_batchPos0based,\
    dna_batch_obs, \
    dna_batch_trinuc = getAllMisChangePosMutPosTuples( targetPdbObject["dna_mis"].numpy(),
                                                                    targetPdbObject[obsField].numpy(),
                                                                    targetPdbObject['dna_lossWeight'].numpy(),
                                                                    targetPdbObject['dna_altAaNum'].numpy(),
                                                                    targetPdbObject['dna_trinuc'].numpy(),
                                                                    changePoss)

    dna_batch_aaRef = np.where(targetPdbObject["feat_refonehot"][dna_batch_changePosMutPos[:, 0], :])[1]

    labelDict_dna = {"obs": dna_batch_obs[:, np.newaxis].astype(np.float32),
                      "obs_sampleWeight": dna_batch_lossWeight[:, np.newaxis].astype(np.float32),
                      "trinuc": dna_batch_trinuc.astype(np.int64),
                      "aaNumAlt": dna_batch_altAaNum.astype(np.int64),
                      "batchPos0based": dna_batchPos0based.astype(np.int64),
                      "aaNumRef": dna_batch_aaRef.astype(np.int64)}

    return labelDict_dna


#@njit()
def getAllMisChangePosMutPosTuples(dna_mis,
                                dna_obs,
                                dna_lossWeight,
                                dna_altAaNum,
                                dna_trinuc,
                                changePoss):
    """ Returns a tuple of 6 numpy arrays that are specific to the subset of
    positions in the protein chosen for the batch.
    - changePosMutPos: (N_mut, 2): a row (u, v) in this array indicates 
          that a mutation exists in position u (1based?) in the protein,
          and v indicates the position of the change in the one-hot codon repr 
          (v ranges from 0 to 11)
    - lossWeight: (N_mut,)
    - aaNum: index of the alternative AA
    - pos0based: (N_mut,): pos0based[i] is the position of the i-th mutation in
          the batch (from 0 to batch_size-1).
    - obs: (N_mut,): (not sure) 0 or 1 depending on whether mutation was
          observed
    - trinuc: (N_mut,): trinucleotide context index from 1 to 192
    """
    #dna_obs_mis = dna_obs * dna_mis

    vec_pos1based_rep = np.repeat(changePoss, 12).reshape(-1, 12).flatten()
    vec_batchpos0based_rep = np.repeat(np.arange(changePoss.shape[0]), 12).reshape(-1, 12).flatten()

    vec_mutPos_rep = np.repeat(np.arange(12), (changePoss.shape[0])).reshape(12, -1).transpose().flatten()

    arr_isGood = (dna_mis[changePoss, :].flatten() == 1) & (dna_obs[changePoss, :].flatten() > -1)

    posMutPosArr = np.stack((vec_pos1based_rep[arr_isGood],
                             vec_mutPos_rep[arr_isGood],
                             vec_batchpos0based_rep[arr_isGood])).transpose()
    np.random.shuffle(posMutPosArr)

    dna_lossWeight_batch = dna_lossWeight[posMutPosArr[:, 0], posMutPosArr[:, 1]]
    dna_altAaNum_batch = dna_altAaNum[posMutPosArr[:, 0], posMutPosArr[:, 1]].astype(np.int64)
    dna_obsMis_batch = dna_obs[posMutPosArr[:, 0], posMutPosArr[:, 1]]
    vec_batchPos0based = posMutPosArr[:, 2 ]
    dna_trinuc_batch = dna_trinuc[posMutPosArr[:, 0], posMutPosArr[:, 1]]

    return posMutPosArr[:, 0:2 ], dna_lossWeight_batch, dna_altAaNum_batch, vec_batchPos0based, dna_obsMis_batch, dna_trinuc_batch

