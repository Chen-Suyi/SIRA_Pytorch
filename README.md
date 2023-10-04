# [ICCV 2023] SIRA-PCR: Sim-to-Real Adaptation for 3D Point Cloud Registration

This is the Pytorch implementation of our ICCV2023 paper [SIRA-PCR](./files/final_version.pdf).

## Introduction
Point cloud registration is essential for many applications.
However, existing real datasets require extremely tedious and costly annotations, yet may not provide accurate camera poses.
For the synthetic datasets, they are mainly object-level, so the trained models may not generalize well to real scenes.
First, we build a synthetic scene-level 3D registration dataset, 
specifically designed with physically-based and random strategies to arrange diverse objects.
Second, we account for variations in different sensing mechanisms and layout placements, then formulate a sim-to-real adaptation framework
with an adaptive re-sample module to simulate patterns in real point clouds.
To our best knowledge, this is the first work that explores sim-to-real adaptation for point cloud registration.
Extensive experiments show the SOTA performance of SIRA-PCR on widely-used indoor and outdoor datasets.  

![image text](./files/pipeline.png)

## Installation
Please use the following command for installation.  
```
# Download the codes
git clone https://github.com/Chen-Suyi/SIRA_Pytorch.git
cd SIRA_PCR

# It is recommended to create a new environment
conda create -n sira python==3.9
conda activate sira

# Install pytorch
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install other packages
pip install -r requirements.txt

# Install dependencies for Geotransformer
cd PCR/extensions
python setup.py build develop
```

## Data Preparation
### 3DMatch
The 3DMatch data can be downloaded from [OverlapPredator](https://github.com/prs-eth/OverlapPredator) by running:
```
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip
```

Please unzip and put it in `SIRA-PCR/PCR/dataset/3DMatch/data`
The data should be organized as follows:
```
--dataset--3DMatch--metadata
                 |--data--train--7-scenes-chess--cloud_bin_0.ply
                 |            |               |--...
                 |            |--...
                 |--test--7-scenes-redkitchen--cloud_bin_0.ply
                 |                          |--...
                 |--...
```

### FlyingShapes
The FlyingShapes data can be downloaded from [here](https://1drv.ms/f/s!Aplan7-8_DG1gVrkdBfWi2xBQZvz?e=GFDqyo).  
Please unzip and put it in `SIRA-PCR/PCR/dataset/FlyingShapes`  
The data should be organized as follows:  
```
--dataset--FlyingShapes--gt.log
                      |--0a71be67-5024-4b9c-a53f-da0aaa294963--depth_exr--cam0_theta0_phi0--pc.ply
                      |                                     |          |                 |--pc_tsdf.ply
                      |                                     |          |--...
                      |                                     |--... 
                      |--...
```
where the point cloud `pc.ply` is used for point cloud registration training, and the point cloud `pc_tsdf.ply` is used as the input of sim-to-real adaptation.  

### Structured3D
The Structured3D data can be downloaded from [here](https://1drv.ms/f/s!Aplan7-8_DG1gVuDz9IZ89QBBLQ9?e=pSHTdu).  
Please unzip and put it in `SIRA-PCR/PCR/dataset/Structured3D`  
The data should be organized as follows:  
```
--dataset--Structured3D--gt.log
                      |--scene_00000--485142_0--pc.ply
                      |            |--...
                      |--...
```
Other datasets will be released soon.  

### SIRA
The SIRA data is the result of sim-to-real adaptation, which was processed by our SIRA from part of  FlyingShapes data. For more details, please refer to our [paper](./files/final_version.pdf).    
The SIRA data can be downloaded from [here](https://1drv.ms/u/s!Aplan7-8_DG1gUCmsMru6F32i6hT?e=Llw1rp).  
Please unzip and put it in `SIRA-PCR/PCR/dataset/SIRA`  
The data should be  organized as follows:  
```
--dataset--SIRA--gt.log
              |--0a71be67-5024-4b9c-a53f-da0aaa294963--depth_exr--cam0_theta0_phi0--pc.ply
              |                                     |          |--...
              |                                     |--... 
              |--...
```

## Pre-trained Weights
We provide pre-trained weights on the [releases](https://github.com/Chen-Suyi/SIRA_Pytorch/releases) page.

- `weights.zip` contains the pre-trained weights for point cloud registration.<br>Please unzip and put it to `SIRA-PCR/PCR/weights`.

- `ckpt.zip` contains the pre-trained weights for sim-to-real adaptation.<br>Please unzip and put it to `SIRA-PCR/SIRA/experiment/synth2real/ckpt`.

## SIRA: Sim-to-Real Adaptation
First, please change the current working directory to `SIRA-PCR/SIRA`.  
```
cd /path/to/SIRA-PCR/SIRA
```
### Adaptation
Use the following command to process FlyingShapes data using SIRA:  
```
python test.py --ckpt_load=model_best_SIRA.pt
```
The results will be in `SIRA-PCR/PCR/dataset/synth2real`.  

### Training
You can also try training your own SIRA model using the following command:  
```
python train.py
```

## Point Cloud Registration
First, please change the current working directory to `SIRA-PCR/PCR`.
```
cd /path/to/SIRA-PCR/PCR
```

### Training
#### Training on synthetic data
Use the following command for training on FlyingShapes.

```
CUDA_VISIBLE_DEVICES=GPUS python train.py --model_dir=./experiment/experiment_geotransformer/train_on_FlyingShapes/
```

Use the following command for training on FlyingShapes and Structured3D.

```
CUDA_VISIBLE_DEVICES=GPUS python train.py --model_dir=./experiment/experiment_geotransformer/train_on_FlyingShapes_Structured3D/
```

Or use the following command to start from pre-trained weights.

```
CUDA_VISIBLE_DEVICES=GPUS python train.py --model_dir=./experiment/experiment_geotransformer/train_on_FlyingShapes_Structured3D/ --resume=./weights/model_best_trained_on_flyingshapes.pth -ow
```

#### Fine-tuning on SIRA
Use the following command for fine-tuning on SIRA.

```
CUDA_VISIBLE_DEVICES=GPUS python train.py --model_dir=./experiment/experiment_geotransformer/finetune_on_SIRA/ --resume=./weights/model_best_trained_on_flyingshapes_structured3d.pth -ow
```
#### Fine-tuning on 3DMatch
Use the following command for fine-tuning on 3DMatch.

```
CUDA_VISIBLE_DEVICES=GPUS python train.py --model_dir=./experiment/experiment_geotransformer/finetune_on_3DMatch/ --resume=./weights/model_best_trained_on_flyingshapes_structured3d.pth -ow
```

### Testing
#### Testing on 3DMatch/3DLoMatch
```
# 3DMatch
CUDA_VISIBLE_DEVICES=0 python test.py --model_dir=./experiment/experiment_geotransformer/finetune_on_3DMatch/ --resume=./weights/model_best_finetuned_on_3dmatch.pth --benchmark=3DMatch
CUDA_VISIBLE_DEVICES=0 python eval.py --model_dir=./experiment/experiment_geotransformer/finetune_on_3DMatch/ --benchmark=3DMatch --method=lgr

# 3DLoMatch
CUDA_VISIBLE_DEVICES=0 python test.py --model_dir=./experiment/experiment_geotransformer/finetune_on_3DMatch/ --resume=./weights/model_best_finetuned_on_3dmatch.pth --benchmark=3DLoMatch
CUDA_VISIBLE_DEVICES=0 python eval.py --model_dir=./experiment/experiment_geotransformer/finetune_on_3DMatch/ --benchmark=3DLoMatch --method=lgr
```

#### Testing on ETH
```
CUDA_VISIBLE_DEVICES=0 python test.py --model_dir=./experiment/experiment_geotransformer/test_on_ETH/ --resume=./weights/model_best_finetuned_on_3dmatch.pth --benchmark=ETH
CUDA_VISIBLE_DEVICES=0 python eval.py --model_dir=./experiment/experiment_geotransformer/test_on_ETH/ --benchmark=ETH --method=lgr
```

## Citation

```
@InProceedings{Chen_2023_ICCV,
    author={Chen, Suyi and Xu, Hao and Li, Ru and Liu, Guanghui and Fu, Chi-Wing and Liu, Shuaicheng},
    title={SIRA-PCR: Sim-to-Real Adaptation for 3D Point Cloud Registration},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month={October},
    year={2023},
    pages={14394-14405}
}
```
