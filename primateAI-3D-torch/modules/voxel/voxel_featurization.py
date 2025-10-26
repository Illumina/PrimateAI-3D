import torch.nn as nn


class VoxelFeaturization(nn.Module):
    def __init__(self, c, grid, pdbDict, multiz):
        super().__init__()

        self.c = c
        self.grid = grid
        self.multiz = multiz
        self.pdbDict = pdbDict

    def featurizeVoxelGrid( self,
                            voxels_argmax_centerIdx,
                            voxels_argmax_batchResIds0Based,
                            prot_feats0based,
                            prot_multizProfiles,
                            voxelFeats_proteinBatch):


        dense_residueFeats = prot_feats0based[ voxels_argmax_batchResIds0Based, : ]

        voxelMultizFeats_proteinBatch_flat = voxelFeats_proteinBatch.view(-1, voxelFeats_proteinBatch.shape[-1])

        residueFeatsStartIdx = 0
        residueFeatsEndIdx = dense_residueFeats.shape[1]

        voxelMultizFeats_proteinBatch_flat[ voxels_argmax_centerIdx, residueFeatsStartIdx:residueFeatsEndIdx ] = dense_residueFeats

        if self.c["input_doMultiz"]:
            multizFeatsStartIdx = residueFeatsEndIdx
            multizFeatsEndIdx = multizFeatsStartIdx + prot_multizProfiles.shape[1]
            voxelMultizFeats_proteinBatch_flat[ voxels_argmax_centerIdx, multizFeatsStartIdx:multizFeatsEndIdx ] = prot_multizProfiles[ voxels_argmax_batchResIds0Based, : ]

        return voxelFeats_proteinBatch
