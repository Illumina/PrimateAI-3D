import torch

def _apply_pairwise_op(op, refTensor, compTensor):
    t1 = torch.unsqueeze(torch.unsqueeze(refTensor, 2), 1)
    t2 = torch.unsqueeze(compTensor, 2)

    res = op(t1, t2)

    return res


def applyPairwiseOp(targetTensor, compRangeLength, op):
    targetTensorDouble = torch.concat([targetTensor, targetTensor], axis=0)

    batchSize = targetTensor.shape[0]

    compTensorIndices = torch.repeat_interleave(torch.unsqueeze(torch.arange(0, compRangeLength), 0), batchSize, dim=0)
    compTensorIndices = compTensorIndices + torch.unsqueeze(torch.arange(0, batchSize), 1)

    tensorDouble_comp = targetTensorDouble[compTensorIndices]

    tensor_pairwise = _apply_pairwise_op(op, targetTensor, tensorDouble_comp)

    tensor_pairwise = tensor_pairwise.view(batchSize, -1)

    return tensor_pairwise


def pairwiseLoss_helper(y_true, y_pred, compRangeLength, mask_value):
    compRangeLength = torch.minimum(torch.tensor(compRangeLength), torch.tensor(y_true.shape[0]))

    y_mask = y_true != mask_value

    y_true_pairwise = applyPairwiseOp(y_true, compRangeLength, torch.subtract)
    y_pred_pairwise = applyPairwiseOp(y_pred, compRangeLength, torch.subtract)
    y_mask_pairwise = applyPairwiseOp(y_mask, compRangeLength, torch.logical_and)

    return y_true_pairwise, y_pred_pairwise, y_mask_pairwise


def clamped_signed_power_helper(x, power=2, clampMin=-5):
    y = torch.sign(x) * torch.pow(torch.abs(x), power)
    return torch.clamp(y, min=clampMin)


def softplus_helper(x, logisticCurvature):
    """
    Softplus is identical (ignoring the threshold for numerical stability) to lossLogistic_helper(-x)
    and torch.log1p(torch.exp(x * logisticCurvature))/ logisticCurvature

    From Pytorch
    SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.
    For numerical stability the implementation reverts to the linear function when input*beta>threshold
    """
    softplus = torch.nn.Softplus(beta=logisticCurvature, threshold=20)
    return softplus(x)


class PairwiseLoss():
    """
    PairwiseLoss supporting two different version power and logistic according to how the disagreement between two scores is penalized
    y_true: target score
    y_pred: model prediction

    Reduces to original PairwiseLogisticLoss 
    by using loss_rank_pairwise_lossStd=False 
    AND loss_rank_pairwise_targetTransform=plus_one
    AND loss_rank_pairwise_type='logistic'

    Shared parameters
    loss_rank_pairwise_compRangeLength: range over which scores are compared
    loss_rank_pairwise_minScoreSep: minimum absolute pairwise difference for targetScores to be considered different (whether to penalize)
    loss_rank_pairwise_lossStd: whether to standardize the input
    loss_rank_pairwise_targetTransform: how to weight the pairwise distance of the target scores

    Power version specific parameters:
    - loss_rank_pairwisePower_clampMin: controls the minimum cutoff for penalizing the disagreement between pairwise distances
    - loss_rank_pairwisePower_signedPower: controls the power for penalizing the disagreement between pairwise distances

    Logistic version specific parameters:
    -loss_rank_pairwiseLogistic_logisticCurvature: controls the softplus shape with small values, i.e. 2.5 being smooth and larger more relu like
    """

    def __init__(self, c):
        self.c = c
        self.loss_weight = c["loss_weight_rank_pairwise" + c["loss_rank_pairwise_type"].capitalize()]

    def calculate(self, y_true, y_pred):

        version = self.c["loss_rank_pairwise_type"].capitalize()
        y_loss_perSample = PairwiseLoss.lossPairwise(y_true,
                                                     y_pred,
                                                     self.c[f"loss_rank_pairwise{version}_compRangeLength"],
                                                     self.c["mask_value"],
                                                     self.c[f"loss_rank_pairwise{version}_minScoreSep"],
                                                     self.c["loss_rank_pairwisePower_signedPower"],
                                                     self.c["loss_rank_pairwisePower_clampMin"],
                                                     self.c[f"loss_rank_pairwiseLogistic_logisticCurvature"],
                                                     version.lower(),
                                                     False,
                                                     self.c[f"loss_rank_pairwise_targetTransform"]
                                                     )

        return y_loss_perSample

    @staticmethod
    def lossPairwise(y_true, y_pred, compRangeLength, mask_value, minScoreSep, signedPower, clampMin, logisticCurvature,
                     loss_rank_pairwise_type, lossStd, targetTransform):

        # the first dimension is treated as a batch dimension, while the pairwise loss is applied across the second dimension
        # standardize input before moving into the pairwise rank calculation 
        # this removes the magnitude dependence of the loss
        if lossStd:
            y_mask = y_true != mask_value  # (i.e. [batch_posPerProt, batch_protsPerBatch] = [250, 20])
            y_true = masked_standardization(y_true, y_mask=y_mask, mask_value=mask_value, dim=None)
            y_pred = masked_standardization(y_pred, y_mask=y_mask, mask_value=mask_value, dim=None)  # mask is implicitly imposed
            y_pred = torch.where(y_mask, y_pred, y_mask)  # reset mask value

        y_true_pairwise, y_pred_pairwise, y_mask_pairwise = pairwiseLoss_helper(y_true, y_pred, compRangeLength, mask_value)
        if targetTransform == "thresholded_weights":
            y_true_pairwise_ceil = y_true_pairwise * torch.where(torch.abs(y_true_pairwise) >= minScoreSep,
                                                                 1,
                                                                 0.0)
            # mask 
            y_true_pairwise_ceil = y_true_pairwise_ceil * y_mask_pairwise
        elif targetTransform == "thresholded_weights_square":
            y_true_pairwise_ceil = torch.sign(y_true_pairwise) * (y_true_pairwise ** 2) * torch.where(torch.abs(y_true_pairwise) >= minScoreSep,
                                                                                                      1,
                                                                                                      0.0)
            # mask 
            y_true_pairwise_ceil = y_true_pairwise_ceil * y_mask_pairwise
        elif targetTransform == 'plus_one':
            y_true_pairwise_ceil = y_true_pairwise >= minScoreSep
            y_true_pairwise_ceil = y_true_pairwise_ceil & y_mask_pairwise
        else:
            raise ValueError(f"Not implemented {targetTransform}")

        y_pred_pairwise_mask = y_pred_pairwise * y_true_pairwise_ceil.float()

        if loss_rank_pairwise_type == "power":
            y_loss = clamped_signed_power_helper(-y_pred_pairwise_mask,
                                                 power=signedPower, clampMin=clampMin)
        elif loss_rank_pairwise_type == "logistic":
            y_loss = softplus_helper(-y_pred_pairwise_mask, logisticCurvature)
        else:
            raise ValueError(f"Not implemented {loss_rank_pairwise_type}")

        if targetTransform == 'plus_one':
            y_loss = y_loss * y_true_pairwise_ceil.float()
        else:
            y_loss = y_loss * y_mask_pairwise

        # summing over the protein dimension
        y_loss_perSample = y_loss.sum(dim=1)
        y_true_pairwise_ceil_perSample = y_true_pairwise_ceil.sum(dim=1)
        y_true_pairwise_ceil_perSample = torch.maximum(torch.tensor(1), y_true_pairwise_ceil_perSample)
        y_loss_perSample = y_loss_perSample / y_true_pairwise_ceil_perSample

        return y_loss_perSample


def masked_standardization(x, y_mask=None, dim=0, mask_value=None):
    """Differentiable masked input standardization"""
    x_centred = x - masked_mean(x, y_mask, dim=dim, keepdim=True)
    x_std = masked_std(x_centred, y_mask, dim=dim, keepdim=True)
    return torch.where(y_mask, x_centred / (x_std + 1e-8), mask_value)


def masked_mean(x, mask=None, dim=1, keepdim=False):
    """Differentiable masked mean"""
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool, device=x.device)
    masked_x = torch.where(mask, x, 0)
    # lower bound the number of masks to 1.0 to avoid undefined terms
    N_mask = torch.clamp(mask.sum(dim=dim, keepdim=True), min=1.0)
    masked_div = masked_x / N_mask
    mean = masked_div.sum(dim=dim, keepdim=keepdim)
    return mean


def masked_std(x, mask=None, dim=-1, keepdim=False):
    """Differentiable masked std"""
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool, device=x.device)

    masked_x = torch.where(mask, x, 0)
    # lower bound the number of masks to 1.0 to avoid undefined terms
    N_mask = torch.clamp(mask.sum(dim=dim, keepdim=True), min=1.0)
    masked_div = masked_x / N_mask
    mean = masked_div.sum(dim=dim, keepdim=True)

    # subtract the mean and square the result
    var = (masked_x - mean) ** 2
    var = torch.where(mask, var, 0)  # need to reset the mask

    mean_var_div = var / N_mask
    mean_var = mean_var_div.sum(dim=dim, keepdim=keepdim)
    # compute the standard deviation (without clamping gives rise to gradient error here due to invalid values))
    std_dev = torch.sqrt(torch.clamp(mean_var, 1e-8))

    return std_dev



class MaskedMSELoss():
    """
    MSE wrapper to match the rank loss format and mask
    """

    def __init__(self, c):
        self.loss = torch.nn.MSELoss(reduce=False)
        self.c = c
        self.mask_value = self.c["mask_value"]
        self.loss_weight = c["loss_weight_rank_mse"]

    def calculate(self, y_true, y_pred):
        y_mask = y_true != self.mask_value
        loss = self.loss(y_pred, y_true)
        return masked_mean(loss, mask=y_mask, dim=None, keepdim=False)



def getRankLossObj(c):
    rankLossObj = None
    if c["loss_rank_type"] == "pairwise":
        rankLossObj = PairwiseLoss(c)
    elif c["loss_rank_type"] == "mse":
        rankLossObj = MaskedMSELoss(c)
    else:
        raise ValueError(c["loss_rank_type"] + " loss_rank_type not implemented")

    return rankLossObj
