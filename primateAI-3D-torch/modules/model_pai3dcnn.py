import torch
import torch.nn as nn

from helper.helper_nn import get_activation_function, ConvolutionalBlock
from layer.layer_voxel import Voxellayer
from helper.helper_multiprot import addScoresToSubprots


class PrimateAI3D(nn.Module):
    def __init__(self, c, pdbDict, multiz):
        super().__init__()

        self.c = c

        self.voxelLayer = Voxellayer(c, pdbDict, multiz)
        self.nFeats = self.voxelLayer.getNFeats()

        print("Working with %d features" % self.nFeats)

        self.cnnLayers = PrimateAI3D_cnn(c, self.nFeats)

        self.diff_species_layer = self.voxelLayer.diff_species_layer

    def forward(self, multiprotein):
        #print("forward1 ", torch.cuda.memory_allocated() / 1e9, "GB; reserved", torch.cuda.memory_reserved() / 1e9, "GB")
        x = self.voxelLayer(multiprotein)

        score_dict = self.cnnLayers(x)

        addScoresToSubprots(multiprotein.protList, score_dict)

        return score_dict


class PrimateAI3D_cnn(nn.Module):

    def __init__(self, c, nFeats):
        super().__init__()

        self.c = c

        self.nFeats = nFeats
        self.number_of_output_channels = self.get_number_of_output_channels()

        self.main_activation_function = get_activation_function( "relu" )
        self.use_kaiming_initialization = c["model_kaimingInitialization"] and not c["model_batchNormAfterConv"]

        self.initialConv = self.getInitialConv()
        self.reduceBlocks = self.getReduceBlocks()
        self.finalLayers = self.getFinalLayers()

    def getInitialConv(self):

        # Initial 1x1x1 convolutional embedding
        conv = nn.Conv3d(
            in_channels=self.nFeats,
            out_channels=self.c["model_voxels_initalFilters"],
            kernel_size=1,
            bias=True,
        )
        if self.use_kaiming_initialization:
            nn.init.kaiming_uniform_(conv.weight)

        bn = nn.BatchNorm3d(self.c["model_voxels_initalFilters"], track_running_stats=False)
        actv = self.main_activation_function

        if self.c["model_batchNormAfterConv"]:
            layers = [conv, bn, actv]
        else:
            layers = [conv, actv, bn]

        embed_conv = nn.Sequential(*layers)

        return embed_conv

    def getReduceBlocks(self):
        # Reduce blocks
        layers = []
        starting_dim = self.c["voxel_nVoxels"][0]
        kernel_size = (3 ,3 ,3)

        layers.append(ConvolutionalBlock(
            in_channels=self.c["model_voxels_initalFilters"],
            out_channels=self.c["model_voxels_mainFilters"],
            kernel_size=kernel_size,
            batch_norm_after_conv=self.c["model_batchNormAfterConv"],
            activation_function=self.main_activation_function,
            kaiming_initialization=self.use_kaiming_initialization
        ))

        starting_dim = starting_dim - 2

        while starting_dim > 1:
            layers.append(ConvolutionalBlock(
                in_channels=self.c["model_voxels_mainFilters"],
                out_channels=self.c["model_voxels_mainFilters"],
                kernel_size=kernel_size,
                batch_norm_after_conv=self.c["model_batchNormAfterConv"],
                activation_function=self.main_activation_function,
                kaiming_initialization=self.use_kaiming_initialization
            ))
            starting_dim = starting_dim - 2

        reduce_blocks = nn.Sequential(*layers)

        return reduce_blocks


    def getFinalLayers(self):
        # Output connector
        layers = []

        fl = nn.Flatten(start_dim=1)

        fc = nn.Linear(self.c["model_voxels_mainFilters"], self.c["model_voxels_finalDenseNrHidden"], bias=True)
        if self.use_kaiming_initialization:
            nn.init.kaiming_uniform_(fc.weight)

        bn = nn.BatchNorm1d(self.c["model_voxels_finalDenseNrHidden"], track_running_stats=False)
        actv = self.main_activation_function

        if self.c["model_batchNormAfterConv"]:
            layers.extend([fl, fc, bn, actv])
        else:
            layers.extend([fl, fc, actv, bn])

        dropout = nn.Dropout(p=0.1)
        fc = nn.Linear(self.c["model_voxels_finalDenseNrHidden"], self.number_of_output_channels)

        actv = get_activation_function(self.c["model_activationFun"])

        layers.extend([dropout, fc, actv])

        output_connector = nn.Sequential(*layers)

        return output_connector

    def get_number_of_input_channels(self):
        return 1

    def get_number_of_output_channels(self):
        return 20

    def forward(self, model_input):

        retDict = {}
        for voxelType in ["full", "jigsaw"]: #"jigsaw",

            #TODO
            x = model_input[voxelType]

            x = self.initialConv(x)
            x = self.reduceBlocks(x)
            x = self.finalLayers(x)

            retDict["scores_%s" % voxelType] = x

        retDict["scores_both"] = torch.stack(list(retDict.values())).mean(dim=0)


        return retDict


