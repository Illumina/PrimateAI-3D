import torch
import torch.nn as nn

from voxel.voxel_grid import Voxelgrid
from voxel.voxel_featurization import VoxelFeaturization
from voxel.voxel_multiProtVoxelization import voxelize
from layer.layer_diffSpeciesPwm import DiffSpeciesPWM
from helper.helper_data import getCombinedNumberOfFeatures


class Voxellayer(nn.Module):
    def __init__(self, c, pdbDict, multiz):
        super().__init__()
        self.grid = Voxelgrid(c)

        self.featurization = VoxelFeaturization(c, self.grid, pdbDict, multiz)
        self.diff_species_layer = None
        if c['input_doMultiz']:
            self.diff_species_layer = DiffSpeciesPWM(c, multiz)
        self.multiz = multiz
        self.pdbDict = pdbDict

        self.c = c

    def getNFeats(self):
        nDistFeats = 1 + (self.c["voxel_nAAs"] * self.c["nAtoms"] )

        nOtherFeats = getCombinedNumberOfFeatures(
            self.pdbDict,
            self.c["model_voxels_extraVoxelFeatures"],
            self.c, 
            self.diff_species_layer,
        )

        nIndicators = 2 #Jigsaw indicators

        print("nDistFeats:", nDistFeats,
              "nOtherFeats:", nOtherFeats,
              "nIndicators:", nIndicators)

        nAllFeats = nDistFeats + nOtherFeats + nIndicators

        return nAllFeats


    def forward(self, multiProt):

        voxels_proteinBatch_allAtom_jigsaw, \
        voxels_proteinBatch_perAA_jigsaw, \
        voxels_argmax_centerIdx_jigsaw, \
        voxels_argmax_batchResIds0Based_jigsaw, \
        voxels_proteinBatch_allAtom_full, \
        voxels_proteinBatch_perAA_full, \
        voxels_argmax_centerIdx_full, \
        voxels_argmax_batchResIds0Based_full, \
        prot_feats0based, \
        prot_multizAlis0based = voxelize(multiProt, self.grid, self.c)

        batchSize = voxels_proteinBatch_allAtom_jigsaw.shape[0]

        #multiz feats
        prot_multizProfile = None
        nMultizFeats = 0
        if self.c["input_doMultiz"]:
            prot_multizProfiles = self.diff_species_layer.process_2d(prot_multizAlis0based)

            prot_multizProfile = torch.concat(list(prot_multizProfiles.values()), dim=1)
            nMultizFeats = self.diff_species_layer.n_out_features


        nOtherFeats = prot_feats0based.shape[1] + nMultizFeats + 2

        voxels_proteinBatch_featGrid_jigsaw = self.getFeatsVoxelsGrid(nOtherFeats, batchSize)
        voxels_proteinBatch_featGrid_jigsaw = self.featurization.featurizeVoxelGrid(
            voxels_argmax_centerIdx_jigsaw,
            voxels_argmax_batchResIds0Based_jigsaw,
            prot_feats0based,
            prot_multizProfile,
            voxels_proteinBatch_featGrid_jigsaw,
        )

        voxels_proteinBatch_featGrid_jigsaw[:, :, :, :, -1] = 1 #Jigsaw indicator
        voxels_proteinBatch_featGrid_jigsaw[:, :, :, :, -2] = 0 #Jigsaw indicator

        voxels_proteinBatch_featGrid_full = voxels_proteinBatch_featGrid_jigsaw.clone()
        #voxels_proteinBatch_featGrid_full = self.getFeatsVoxelsGrid(nOtherFeats, batchSize)
        voxels_proteinBatch_featGrid_full = self.featurization.featurizeVoxelGrid(
            voxels_argmax_centerIdx_full,
            voxels_argmax_batchResIds0Based_full,
            prot_feats0based,
            prot_multizProfile,
            voxels_proteinBatch_featGrid_full,
        )



        voxels_proteinBatch_featGrid_full[:, :, :, :, -1] = 0
        voxels_proteinBatch_featGrid_full[:, :, :, :, -2] = 1 #PAI Indicator


        # print(voxels_proteinBatch_allAtom_jigsaw.shape)
        # print(voxels_proteinBatch_perAA_jigsaw.shape)
        # print(voxels_proteinBatch_allAtom_full.shape)
        # print(voxels_proteinBatch_perAA_full.shape)

        voxels_jigsaw = torch.cat([voxels_proteinBatch_allAtom_jigsaw,
                                   voxels_proteinBatch_perAA_jigsaw,
                                   voxels_proteinBatch_featGrid_jigsaw], dim=-1)

        voxels_full = torch.cat([  voxels_proteinBatch_allAtom_full,
                                   voxels_proteinBatch_perAA_full,
                                   voxels_proteinBatch_featGrid_full], dim=-1)

        #NHWC --> NCHW
        voxels_jigsaw = torch.transpose(voxels_jigsaw, 1, -1)
        voxels_full = torch.transpose(voxels_full, 1, -1)

        returnDict = {"jigsaw": voxels_jigsaw, "full": voxels_full}


        return returnDict

    def getFeatsVoxelsGrid(self, nFeats, batchSize):

        outputDims = [       batchSize,
                             self.grid.nVoxels_oneDim,
                             self.grid.nVoxels_oneDim,
                             self.grid.nVoxels_oneDim,
                             nFeats]

        voxelFeats_proteinBatch = torch.zeros(outputDims, dtype=torch.float32, device=self.c["torch_device"])

        return voxelFeats_proteinBatch

