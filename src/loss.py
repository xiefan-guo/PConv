import torch
import torch.nn as nn

import src.config as config
from src.utils import gram_matrix


# --------
# tv loss
# --------
def total_variation_loss(image):
    # ---------------------------------------------------------------
    # shift one pixel and get difference (for both x and y direction)
    # ---------------------------------------------------------------
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))

    return loss


def generator_loss(inputs, masks, outputs, ground_truths, extractor):

    l1 = nn.L1Loss()

    comp = masks * inputs + (1 - masks) * outputs

    # ---------
    # hole loss
    # ---------
    loss_hole = l1((1 - masks) * outputs, (1 - masks) * ground_truths)

    # ----------
    # valid loss
    # ----------
    loss_valid = l1(masks * outputs, masks * ground_truths)

    if outputs.size(1) == 3:
        feat_comp = extractor(comp)
        feat_output = extractor(outputs)
        feat_gt = extractor(ground_truths)
    elif outputs.size(1) == 1:
        feat_comp = extractor(torch.cat([comp] * 3, dim=1))
        feat_output = extractor(torch.cat([outputs] * 3, dim=1))
        feat_gt = extractor(torch.cat([ground_truths] * 3, dim=1))
    else:
        raise ValueError('only gray rgb')

    # ---------------
    # perceptual loss
    # ---------------
    loss_perceptual = 0.0
    for i in range(3):
        loss_perceptual += l1(feat_output[i], feat_gt[i])
        loss_perceptual += l1(feat_comp[i], feat_gt[i])

    # ----------
    # style loss
    # ----------
    loss_style = 0.0
    for i in range(3):
        loss_style += l1(gram_matrix(feat_output[i]),
                              gram_matrix(feat_gt[i]))
        loss_style += l1(gram_matrix(feat_comp[i]),
                              gram_matrix(feat_gt[i]))

    # -------
    # tv loss
    # -------
    loss_tv = total_variation_loss((1 - masks) * outputs)

    total_loss = loss_hole * config.HOLE_LOSS + loss_valid * config.VALID_LOSS + \
                 loss_perceptual * config.PERCEPTUAL_LOSS + loss_style * config.STYLE_LOSS +\
                 loss_tv * config.TOTAL_VARIATION_LOSS

    return total_loss, [loss_hole, loss_valid, loss_perceptual, loss_style, loss_tv]

