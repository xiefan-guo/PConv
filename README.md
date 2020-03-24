# PConv

PyTorch implementation of "Image Inpainting for Irregular Holes Using Partial Convolutions (ECCV 2018)" [[Paper]](https://arxiv.org/abs/1804.07723).

**Authors**: _Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, Bryan Catanzaro_

## Prerequisites

* Python 3
* PyTorch 1.0
* NVIDIA GPU + CUDA cuDNN

## Installation

* Clone this repo:

```
git clone https://github.com/Xiefan-Guo/PConv.git
cd PConv
```

## Usage

### Training

To train the PUNet model:

```
python train.py \ 
    --image_root [path to input image directory] \ 
    --mask_root [path to masks directory]
```

### Evaluating

To evaluate the model:

```
python eval.py \
    --pre_trained [path to checkpoints] \
    --image_root [path to input image directory] \ 
    --mask_root [path to masks directory]
```