import torch
import torch.nn as nn


def middle_of_window(c):
    win_sz = c["model_sequence_windowSize"]
    return (win_sz - 1) // 2


class ResidualBlock(nn.Module):
    """ Residual block. Very simple, but useful to code it as a layer instead
    of in the forward, in order to visualize the residual connections when
    printing the model.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x

class ConvolutionalBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            batch_norm_after_conv=False,
            activation_function=nn.ReLU,
            include_bias=True,
            kaiming_initialization=True,
            number_of_residual_units=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.batch_norm_after_conv = batch_norm_after_conv
        self.activation_function = activation_function
        self.include_bias = include_bias
        self.kaiming_initialization = kaiming_initialization
        self.number_of_residual_units = number_of_residual_units

        self.conv_reduce = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            bias=self.include_bias,
        )

        if self.kaiming_initialization:
            nn.init.kaiming_uniform_(self.conv_reduce.weight)

        bn = nn.BatchNorm3d(self.out_channels, track_running_stats=False)
        actv = self.activation_function

        if self.batch_norm_after_conv:
            self.outer_layers = nn.Sequential(bn, actv)
        else:
            self.outer_layers = nn.Sequential(actv, bn)

    def forward(self, x):

        x = self.conv_reduce(x)

        if self.number_of_residual_units:
            x = self.inner_layers(x)

        x = self.outer_layers(x)

        return x

class ConvolutionalBlock1D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            batch_norm_after_conv=False,
            activation_function=nn.ReLU,
            include_bias=True,
            kaiming_initialization=True,
            padding=0,
            residual=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.batch_norm_after_conv = batch_norm_after_conv
        self.include_bias = include_bias
        self.kaiming_initialization = kaiming_initialization
        self.residual = residual

        conv_reduce = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            bias=self.include_bias,
            padding=padding,
        )

        if self.kaiming_initialization:
            nn.init.kaiming_uniform_(conv_reduce.weight)

        bn = nn.BatchNorm1d(self.out_channels, track_running_stats=False)
        layers = [conv_reduce]

        if self.batch_norm_after_conv:
            layers += [bn, activation_function]
        else:
            layers += [activation_function, bn]
        if self.residual:
            self.sequential = ResidualBlock(nn.Sequential(*layers))
        else:
            self.sequential = nn.Sequential(*layers)


    def forward(self, x):
        return self.sequential(x)


def get_activation_function(function_name):

    function_name = function_name.lower()

    if function_name == 'relu':
        function = nn.ReLU()
    elif function_name == 'tanh':
        function = nn.Tanh()
    elif function_name == 'sigmoid':
        function = nn.Sigmoid()
    elif function_name == 'elu':
        function = nn.ELU()
    elif function_name == 'linear':
        function = nn.Identity()
    elif function_name == 'softmax':
        function = nn.Softmax(dim=-1)


    # elif function_name == 'exp':
    #     function = nn.ex

    return function


class WindowCNN(nn.Module):
    """ Given an input of size (batch_size, nFeats, window_size), processes
    it with a CNN and returns (batch_size, nOutputChannels).
    """
    def __init__(self, c, nFeats, nOutputChannels):
        super().__init__()
        self.c = c
        self.debug = None
        self.number_output_channels = nOutputChannels
        self.main_activation_function = get_activation_function("relu")
        self.use_kaiming_initialization = c["model_kaimingInitialization"] and not c["model_batchNormAfterConv"]
        self.final_aggregation = self.c['model_sequence_finalAggregation']
        if self.final_aggregation == 'middle':
            assert(self.c['model_sequence_finalDenseNrHidden'] ==
                   self.c['model_sequence_mainFilters'])
        elif self.final_aggregation == 'fc':
            assert(self.c['model_sequence_mainKernelSize'] == 1)
        else:
            raise ValueError(f"Unknown model_sequence_finalAggregation={self.final_aggregation} value")

        self.middle_position = middle_of_window(self.c)

        self.initialConv = self.getInitialConv(nFeats)
        self.mainBlocks = self.getMainBlocks()
        self.finalLayers = self.getFinalLayers()

    def forward(self, x):
        x = self.initialConv(x)
        if self.c["model_sequence_mainBlocks"] > 0:
            x = self.mainBlocks(x)
        if self.final_aggregation == 'middle':
            x = x[:, :, self.middle_position]
        return self.finalLayers(x)


    def getInitialConv(self, nFeats):
        conv = nn.Conv1d(
            in_channels=nFeats,
            out_channels=self.c["model_sequence_initalFilters"],
            kernel_size=self.c["model_sequence_kernelSize"],
            padding='same',
            bias=True,
        )
        if self.use_kaiming_initialization:
            nn.init.kaiming_uniform_(conv.weight)
        bn = nn.BatchNorm1d(self.c["model_sequence_initalFilters"], track_running_stats=False)
        actv = self.main_activation_function
        if self.c["model_batchNormAfterConv"]:
            layers = [conv, bn, actv]
        else:
            layers = [conv, actv, bn]
        embed_conv = nn.Sequential(*layers)
        return embed_conv

    def getMainBlocks(self):
        kernel_size = self.c['model_sequence_mainKernelSize']
        # TODO why are there params model_initalFilters as well? (ie identical
        # without _sequence???
        padding = kernel_size // 2
        layers = [ConvolutionalBlock1D(
            in_channels=self.c["model_sequence_initalFilters"],
            out_channels=self.c["model_sequence_mainFilters"],
            kernel_size=kernel_size,
            batch_norm_after_conv=self.c["model_batchNormAfterConv"],
            activation_function=self.main_activation_function,
            kaiming_initialization=self.use_kaiming_initialization,
            padding=padding,
            residual=self.c['model_residualConnections'],
        )]
        for _ in range(1, self.c["model_sequence_mainBlocks"]):
            layers.append(ConvolutionalBlock1D(
                in_channels=self.c["model_sequence_mainFilters"],
                out_channels=self.c["model_sequence_mainFilters"],
                kernel_size=kernel_size,
                batch_norm_after_conv=self.c["model_batchNormAfterConv"],
                activation_function=self.main_activation_function,
                kaiming_initialization=self.use_kaiming_initialization,
                padding=padding,
                residual=self.c['model_residualConnections'],
            ))
        reduce_blocks = nn.Sequential(*layers)
        return reduce_blocks

    def getFinalLayers(self):
        # Output connector
        layers = []

        if self.final_aggregation == 'fc':
            fl = nn.Flatten(start_dim=1)

            inputUnits = self.c["model_sequence_windowSize"] * (self.c["model_sequence_initalFilters"] if (self.c["model_sequence_mainBlocks"] == 0) else self.c["model_sequence_mainFilters"])

            fc = nn.Linear(inputUnits, self.c["model_sequence_finalDenseNrHidden"], bias=True)
            if self.use_kaiming_initialization:
                nn.init.kaiming_uniform_(fc.weight)

            bn = nn.BatchNorm1d(self.c["model_sequence_finalDenseNrHidden"], track_running_stats=False)
            actv = self.main_activation_function

            if self.c["model_batchNormAfterConv"]:
                layers.extend([fl, fc, bn, actv])
            else:
                layers.extend([fl, fc, actv, bn])
            layers.append(nn.Dropout(p=0.1))

        fc = nn.Linear(self.c["model_sequence_finalDenseNrHidden"],
                       self.number_output_channels)

        actv = get_activation_function(self.c["model_activationFun"])

        layers.extend([fc, actv])

        output_connector = nn.Sequential(*layers)

        return output_connector


