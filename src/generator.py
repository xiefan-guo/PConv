import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.utils import weights_init


# ------------
# Partial Conv
# ------------
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask=None):

        if mask is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
            self.update_mask.to(input)
            self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


# --------------------------
# PConv-BatchNorm-Activation
# --------------------------
class PConvBNActiv(nn.Module):

    def __init__(self, in_channels, out_channels, bn=True, sample='none-3', activ='relu', bias=False):
        super(PConvBNActiv, self).__init__()

        if sample == 'down-7':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=bias, multi_channel = True)
        elif sample == 'down-5':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=bias, multi_channel = True)
        elif sample == 'down-3':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias, multi_channel = True)
        else:
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias, multi_channel = True)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, images, masks):

        images, masks = self.conv(images, masks)
        if hasattr(self, 'bn'):
            images = self.bn(images)
        if hasattr(self, 'activation'):
            images = self.activation(images)

        return images, masks


# ------------
# Double U-Net
# ------------
class PUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, up_sampling_node='nearest', init_weights=True):
        super(PUNet, self).__init__()

        self.freeze_ec_bn = False
        self.up_sampling_node = up_sampling_node

        self.ec_images_1 = PConvBNActiv(in_channels, 64, bn=False, sample='down-7')
        self.ec_images_2 = PConvBNActiv(64, 128, sample='down-5')
        self.ec_images_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_images_4 = PConvBNActiv(256, 512, sample='down-3')
        self.ec_images_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_images_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_images_7 = PConvBNActiv(512, 512, sample='down-3')

        self.dc_images_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_4 = PConvBNActiv(512 + 256, 256, activ='leaky')
        self.dc_images_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_images_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_images_1 = PConvBNActiv(64 + out_channels, out_channels, bn=False, sample='none-3', activ=None, bias=True)

        self.tanh = nn.Tanh()

        if init_weights:
            self.apply(weights_init())

    def forward(self, input_images, input_masks):

        ec_images = {}

        ec_images['ec_images_0'], ec_images['ec_images_masks_0'] = input_images, input_masks
        ec_images['ec_images_1'], ec_images['ec_images_masks_1'] = self.ec_images_1(input_images, input_masks)
        ec_images['ec_images_2'], ec_images['ec_images_masks_2'] = self.ec_images_2(ec_images['ec_images_1'], ec_images['ec_images_masks_1'])
        ec_images['ec_images_3'], ec_images['ec_images_masks_3'] = self.ec_images_3(ec_images['ec_images_2'], ec_images['ec_images_masks_2'])
        ec_images['ec_images_4'], ec_images['ec_images_masks_4'] = self.ec_images_4(ec_images['ec_images_3'], ec_images['ec_images_masks_3'])
        ec_images['ec_images_5'], ec_images['ec_images_masks_5'] = self.ec_images_5(ec_images['ec_images_4'], ec_images['ec_images_masks_4'])
        ec_images['ec_images_6'], ec_images['ec_images_masks_6'] = self.ec_images_6(ec_images['ec_images_5'], ec_images['ec_images_masks_5'])
        ec_images['ec_images_7'], ec_images['ec_images_masks_7'] = self.ec_images_7(ec_images['ec_images_6'], ec_images['ec_images_masks_6'])

        # --------------
        # images decoder
        # --------------
        dc_images, dc_images_masks = ec_images['ec_images_7'], ec_images['ec_images_masks_7']
        for _ in range(7, 0, -1):

            ec_images_skip = 'ec_images_{:d}'.format(_ - 1)
            ec_images_masks = 'ec_images_masks_{:d}'.format(_ - 1)
            dc_conv = 'dc_images_{:d}'.format(_)

            dc_images = F.interpolate(dc_images, scale_factor=2, mode=self.up_sampling_node)
            dc_images_masks = F.interpolate(dc_images_masks, scale_factor=2, mode=self.up_sampling_node)

            dc_images = torch.cat((dc_images, ec_images[ec_images_skip]), dim=1)
            dc_images_masks = torch.cat((dc_images_masks, ec_images[ec_images_masks]), dim=1)

            dc_images, dc_images_masks = getattr(self, dc_conv)(dc_images, dc_images_masks)

        outputs = self.tanh(dc_images)

        return outputs

    def train(self, mode=True):

        super().train(mode)

        if self.freeze_ec_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'ec' in name:
                    module.eval()

