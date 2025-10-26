import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from collections import defaultdict
import torch.nn.functional as F


def create_multiz_filter(n_species, n_out_channels, requires_grad):
    conv = nn.Conv1d(
        in_channels=n_species,
        out_channels=n_out_channels,
        kernel_size=1,  # if you want to make it a parameter, make sure
        # batching is done differently in PAI3D: right now different
        # subproteins are just concatenated.
        bias=False,
    )
    conv.weight.requires_grad = requires_grad
    # self.conv.weight.data: (n_out_channels, n_species, kernel_size)
    conv.weight.data[:] = 1 / float(n_species)
    return conv


class NormalizedPWMTransform(nn.Module):
    def __init__(self, n_species, requires_grad):
        super().__init__()
        data = torch.ones((n_species,), dtype=torch.float) 
        self.weights = nn.Parameter(data, requires_grad=requires_grad)

    def forward(self, one_hot):
        norm_weights = F.softmax(self.weights, dim=0)
        norm_alis0based_flat = one_hot * norm_weights.view((1, -1, 1))
        profile = torch.sum(norm_alis0based_flat, dim=1, keepdim=True)  # (batch_size*win_size, 1, n_aas)
        return profile


class DiffSpeciesPWM(nn.Module):
    """ A layer that transforms aligned MSA sequences into a PWM
    using a learned weighted sum.
    """
    def __init__(self, c, multiz):
        super().__init__()
        assert(multiz is not None)
        self.name_MSA_fields = c['input_featsDiffMultiz']
        n_pwm_features = len(self.name_MSA_fields)
        assert(n_pwm_features > 0)
        unused_extra_inputs = (c['multiz_featGaps'],
                               c['multiz_featStop'], c['multiz_featGlobal'])
        if any(unused_extra_inputs):
            raise NotImplementedError()
        self.n_species = multiz.nSpecies
        n_out_channels = c['model_nMultizFilters']
        self.n_aas = multiz.n_aas
        self.normalize_weights = c['multiz_normalizeWeights']
        if self.normalize_weights:
            assert(n_out_channels == 1)
        pwmTransform_list = []
        for i in range(n_pwm_features):
            if self.normalize_weights:
                pwmTransform = NormalizedPWMTransform(self.n_species, c['multiz_trainLayer'])
            else:
                pwmTransform = create_multiz_filter(self.n_species, n_out_channels, c['multiz_trainLayer'])
            pwmTransform_list.append(pwmTransform)
        self.pwmTransforms = torch.nn.ModuleList(pwmTransform_list)
        self.device = c['torch_device']

        self.n_out_features = n_out_channels * self.n_aas


    def compute_differentiable_pwm(self, pwmTransform, alis0based):
        """ Args:
        - alis0based: integers from 0 to multiz.n_aas denoting AAs or gaps, ...
          can be a (n_prot, max_prot_len, n_species) tensor or (batch, n_species) 
        """
        n_species = alis0based.size(-1)
        assert(n_species == self.n_species)
        alis0based_flat = alis0based.view((-1, n_species)).long()
        one_hot = nn.functional.one_hot(alis0based_flat, num_classes = self.n_aas).float()
        profile = pwmTransform(one_hot)  # (batch_size*win_size, n_out_channels, n_aas)
        profile = profile.view(alis0based.size()[:-1] + (self.n_out_features,))
        return profile

    def get_weights_multiz(self):
        if not self.normalize_weights:
            weights = [conv.weight.data for conv in self.pwmTransforms]
        else:
            weights = [F.softmax(conv.weights.data, dim=0) for conv in self.pwmTransforms]
        return zip(self.name_MSA_fields, weights)

    def forward(self, multiprot):
        """ Given a multiprot, transform the raw MSAs into PWMs and set the value 
        to subprot object
        """
        d_all = defaultdict(list)
        for subprot in multiprot.protList:
            for field in self.name_MSA_fields:
                tensor = subprot.targetPdbObject[field].float().to(self.device)
                d_all[field].append(tensor)

        outputs = {}
        for conv, (name_MSA, list_tensors) in zip(self.pwmTransforms, d_all.items()):
            lengths = torch.as_tensor([l.size(0) for l in list_tensors])
            padded_MSA = pad_sequence(list_tensors, batch_first=True)
            padded_MSA = padded_MSA.float().to(self.device)
            PWM = self.compute_differentiable_pwm(conv, padded_MSA)
            outputs[name_MSA] = (PWM, lengths)

        for i, subprot in enumerate(multiprot.protList):
            subprot.dict_diff_PWMs = {}

        for field in self.name_MSA_fields:
            PWM, lengths = outputs[field]
            unpadded_sequences = unpad_sequence(PWM, lengths, batch_first=True)
            for i, subprot in enumerate(multiprot.protList):
                 subprot.dict_diff_PWMs[field] = unpadded_sequences[i]

    def process_2d(self, dict_2d):
        outputs = {}
        for conv, (name_MSA, MSA) in zip(self.pwmTransforms, dict_2d.items()):
            PWM = self.compute_differentiable_pwm(conv, MSA)
            outputs[name_MSA] = PWM
        return outputs
