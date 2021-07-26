# Unsupervised 3D Human Mesh Recovery from Noisy Point Clouds

[Xinxin Zuo](https://sites.google.com/site/xinxinzuohome/), [Sen Wang](https://sites.google.com/site/senwang1312home/), [Minglun Gong](http://www.socs.uoguelph.ca/~minglun/), [Li Cheng](http://www.ece.ualberta.ca/~lcheng5/)

[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/pdf/2107.07539)


## Prerequisites
We have tested the code on Ubuntu 18.04/20.04 with CUDA 10.2

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do is to use the [anaconda](https://www.anaconda.com/).

You can create an anaconda environment called `fit3d` using
```
conda env create -f environment.yaml
conda activate fit3d
```

## Download SMPL models
Download [SMPL Female and Male](https://smpl.is.tue.mpg.de/) and [SMPL Netural](https://smplify.is.tue.mpg.de/), and rename the files and extract them to `<current directory>/smpl_models/smpl/`, eventually, the `<current directory>/smpl_models` folder should have the following structure:
   ```
   smpl_models
    └-- smpl
    	└-- SMPL_FEMALE.pkl
		└-- SMPL_MALE.pkl
		└-- SMPL_NEUTRAL.pkl
   ```   

## Download pre-trained models
1. Download two weights (point cloud and depth) from: [Point Cloud](https://drive.google.com/file/d/17MpUwC4fMVoEF3VBzCX82NgLizxlZXEH/view?usp=sharing)  and [Depth](https://drive.google.com/file/d/1kbktLqVWEb-Hsbs-JxfcM7QP1mysOHvo/view?usp=sharing)
2. Put the downloaded weights in `<current directory>/pretrained/`

## Demo
### Demo for whole point cloud
python generate_pt.py --filename ./demo/demo_pt/00010805.ply --gender female
### Demo for partial point cloud/depth
python generate_depth.py --filename ./demo/demo_depth/shortshort_flying_eagle.000075_depth.ply --gender male


## Citation
If you find this project useful for your research, please consider citing:
```
@article{zuo2021unsupervised,
  title={Detailed human shape estimation from a single image by hierarchical mesh deformation},
  author={Zuo, Xinxin and Wang, Sen and Gong, Minglun and Cheng, Li},
  year={2021}
}
```

## References
We indicate if a function or script is borrowed externally inside each file. Here are some great resources we 
benefit:

- Shape/Pose prior and some functions are borrowed from [VIBE](https://github.com/mkocabas/VIBE).
- SMPL models and layer is from [SMPL-X model](https://github.com/vchoutas/smplx).
- Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR).
- Some functions are borrowed from [pointnet-pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
- CAPE dataset for training [CAPE](https://cape.is.tue.mpg.de/）.
- CMU Panoptic Studio dataset for training [CMU Panoptic](http://domedb.perception.cs.cmu.edu/).

