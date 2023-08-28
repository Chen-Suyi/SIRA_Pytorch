# [ICCV 2023] SIRA-PCR: Sim-to-Real Adaptation for 3D Point Cloud Registration

This is the Pytorch implementation of our ICCV2023 paper [SIRA-PCR](./files/final_version.pdf).

## Data Preparation
Our dataset will be released soon.

## Pre-trained Weights
Our pre-trained weights will be released soon.

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
