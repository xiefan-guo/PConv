import os
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.dataset import ImageDataset
from src.generator import PUNet
from src.evaluate import compute_psnr, compute_ssim
from src import config

parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=0, help='workers for dataloader')
parser.add_argument('--pre_trained', type=str, default='',
                    help='pre-trained models for fine-tuning')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--load_size', type=int, default=350, help='image loading size')
parser.add_argument('--crop_size', type=int, default=256, help='image training size')
parser.add_argument('--image_root', type=str, default='')
parser.add_argument('--mask_root', type=str, default='')
parser.add_argument('--result_root', type=str, default='results/eval/default', help='train result')
parser.add_argument('--number_eval', type=int, default=10, help='number of batches eval')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
is_cuda = torch.cuda.is_available()
if is_cuda:
    print('Cuda is available!')
    cudnn.benchmark = True

if not os.path.exists(args.result_root):
    os.makedirs(args.result_root)

image_dataset = ImageDataset(
    args.image_root, args.mask_root, (args.load_size, args.load_size), (args.crop_size, args.crop_size)
)
data_loader = DataLoader(
    image_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=False
)
print(len(data_loader))
generator = PUNet(3, 3)

if args.pre_trained != '':
    generator.load_state_dict(torch.load(args.pre_trained)['generator'])
else:
    print('Please provide pre-trained model!')

for param in generator.parameters():
    param.requires_grad = False

if is_cuda:
    generator = generator.cuda()

print('start eval...')

sum_psnr = 0.0
sum_ssim = 0.0
l1_loss = 0.0
sum_time = 0.0
count = 0


while True:

    if count >= args.number_eval:
        break

    generator.eval()
    for _, (input_images, ground_truths, masks) in enumerate(data_loader):

        if count >= args.number_eval:
            break

        count = count + 1
        if is_cuda:
            input_images, ground_truths, masks = \
                input_images.cuda(), ground_truths.cuda(),  masks.cuda()

        start_time = time.time()
        outputs = generator(input_images, masks)
        end_time = time.time()
        sum_time = sum_time + (end_time - start_time)
        print('time: ', end_time - start_time)
        # canny edge detection

        outputs = outputs.data.cpu()
        ground_truths = ground_truths.data.cpu()
        masks = masks.data.cpu()

        damaged_images = ground_truths * masks + (1 - masks)
        outputs_comps = ground_truths * masks + outputs * (1 - masks)

        psnr = compute_psnr(ground_truths, outputs_comps)
        sum_psnr += psnr
        print(count, ' psnr: ', psnr)

        ssim = compute_ssim(ground_truths * 255, outputs_comps * 255).item()
        sum_ssim += ssim
        print(count, ' ssim: ', ssim)

        l1 = nn.L1Loss()(ground_truths, outputs_comps).item()
        l1_loss += l1
        print(count, ' l1_loss: ', l1)

        sizes = ground_truths.size()
        bound = min(5, sizes[0])
        save_images = torch.Tensor(sizes[0] * 8, sizes[1], sizes[2], sizes[3])

        # -------------
        # simple sample
        # -------------

        save_images_5 = torch.Tensor(sizes[0] * 5, sizes[1], sizes[2], sizes[3])
        for i in range(sizes[0]):
            save_images_5[5 * i] = 1 - masks[i]
            save_images_5[5 * i + 1] = damaged_images[i]
            save_images_5[5 * i + 2] = outputs[i]
            save_images_5[5 * i + 3] = outputs_comps[i]
            save_images_5[5 * i + 4] = ground_truths[i]

        save_image(save_images_5, os.path.join(args.result_root, '{:05d}.png'.format(count)), nrow=5)


print('count: ', count)
print('avgrage l1 loss: ', l1_loss / count)
print('average psnr: ', sum_psnr / count)
print('average ssim: ', sum_ssim / count)
print('average time cost: ', sum_time / count)

print('complete the eval.')
