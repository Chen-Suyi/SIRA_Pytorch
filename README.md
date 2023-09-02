# [ICCV 2023] SIRA-PCR: Sim-to-Real Adaptation for 3D Point Cloud Registration

This is the Pytorch implementation of our ICCV2023 paper [SIRA-PCR](./files/final_version.pdf).

## Data Preparation
The 3DMatch data can be downloaded from [OverlapPredator](https://github.com/prs-eth/OverlapPredator) by running:
```
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip
```

Please unzip and put it in `SIRA-PCR/PCR/dataset/3DMatch/data`
The data should be organized as follows:
```
--dataset--3DMatch--metadata
                 |--data--train--7-scenes-chess--cloud_bin_0.ply
                       |      |               |--...
                       |      |--...
                       |--test--7-scenes-redkitchen--cloud_bin_0.ply
                             |                    |--...
                             |--...
```

Other datasets will be released soon.

## Pre-trained Weights
We provide pre-trained weights on the [releases](https://github.com/Chen-Suyi/SIRA_Pytorch/releases) page.

- `weights.zip` contains the pre-trained weights for point cloud registration.<br>Please unzip and put it to `SIRA-PCR/PCR/weights`.

- `ckpt.zip` contains the pre-trained weights for sim-to-real adaptation.<br>Please unzip and put it to `SIRA-PCR/SIRA/experiment/synth2real/ckpt`.

## Training
### Training on synthetic data
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

### Fine-tuning on SIRA
Use the following command for fine-tuning on SIRA.

```
CUDA_VISIBLE_DEVICES=GPUS python train.py --model_dir=./experiment/experiment_geotransformer/finetune_on_SIRA/ --resume=./weights/model_best_trained_on_flyingshapes_structured3d.pth -ow
```
### Fine-tuning on 3DMatch
Use the following command for fine-tuning on 3DMatch.

```
CUDA_VISIBLE_DEVICES=GPUS python train.py --model_dir=./experiment/experiment_geotransformer/finetune_on_3DMatch/ --resume=./weights/model_best_trained_on_flyingshapes_structured3d.pth -ow
```

## Testing
### Testing on 3DMatch/3DLoMatch
```
# 3DMatch
CUDA_VISIBLE_DEVICES=0 python test.py --model_dir=./experiment/experiment_geotransformer/finetune_on_3DMatch/ --resume=./weights/model_best_finetuned_on_3dmatch.pth --benchmark=3DMatch
CUDA_VISIBLE_DEVICES=0 python eval.py --model_dir=./experiment/experiment_geotransformer/finetune_on_3DMatch/ --benchmark=3DMatch --method=lgr

# 3DLoMatch
CUDA_VISIBLE_DEVICES=0 python test.py --model_dir=./experiment/experiment_geotransformer/finetune_on_3DMatch/ --resume=./weights/model_best_finetuned_on_3dmatch.pth --benchmark=3DLoMatch
CUDA_VISIBLE_DEVICES=0 python eval.py --model_dir=./experiment/experiment_geotransformer/finetune_on_3DMatch/ --benchmark=3DLoMatch --method=lgr
```

### Testing on ETH
```
CUDA_VISIBLE_DEVICES=0 python test.py --model_dir=./experiment/experiment_geotransformer/test_on_ETH/ --resume=./weights/model_best_finetuned_on_3dmatch.pth --benchmark=ETH
CUDA_VISIBLE_DEVICES=0 python eval.py --model_dir=./experiment/experiment_geotransformer/test_on_ETH/ --benchmark=ETH --method=lgr
```

## Citation

```
@InProceedings{SIRA_2023_ICCV,
    author={Chen, Suyi and Xu, Hao and Li, Ru and Liu, Guanghui and Fu, Chi-Wing and Liu, Shuaicheng},
    title={SIRA-PCR: Sim-to-Real Adaptation for 3D Point Cloud Registration},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month={October},
    year={2023},
}
```
