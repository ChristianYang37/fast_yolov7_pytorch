# Fast_Yolov7_Pytorch🎉️🎉️🎉️

***<center>Use SOTA Pruning and Quant Algorithm to Build Your Faster Yolov7🚀️</center>***

## Guide

* [Installation](#index1)
* [Quick Start](#index2)
* [References](#index3)
* [Contact Me](#index4)

## <a id="index1">Installation</a>：

```commandline
git clone https://github.com/ChristianYang37/fast_yolov7_pytorch.git
```

```commandline
pip install -r requirements.txt
```

If you can't install torch_pruning, please do as follow

```commandline
git clone https://github.com/VainF/Torch-Pruning.git
cd Torch-Pruning
python setup.py install
```

For pytorch yolov7 state dicts, [click here](https://github.com/WongKinYiu/yolov7#transfer-learning) to download.

## <a id="index2">Quick Start</a>：

### Pruning & Training

```commandline
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml --sparsity 0.01 --num_epoch_to_prune 4 --prune_nore L2
```

if you want prune model without training, you can just set `epochs` = 0

#### Opts

`sparsity`: the sparsity of pruning

`num_epoch_to_prune`: prune model after `num_epoch_to_prune` times finetune

`prune_norm`: L1 or L2

#### Some Results

Prune `yolov7_training.pt` On COCO128.yaml (without finetune)


| Sparsity | Macs       | num_params | mAP@.5  | mAP@.0:.95 |
| -------- | ---------- | ---------- | ------- | ---------- |
| 0.005    | 6379356844 | 37115689   | 0.791   | 0.541      |
| 0.007    | 6373571463 | 37033908   | 0.783   | 0.515      |
| 0.01     | 6324846255 | 36735256   | 0.758   | 0.508      |
| 0.02     | 6187011754 | 35974768   | 0.615   | 0.38       |
| 0.05     | 5820065160 | 33891742   | 0.25    | 0.123      |
| 0.1      | 5237469860 | 30417686   | 0.00056 | 0.000102   |

So, for more efficient pruning, we suggest you set `--sparsity` 0.01 or 0.02, and set `--num_batch_to_prune` big enough to make sure the model has fitted the data before you prune it.

### Quantization

```commandline
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml --method static --epochs 0
```

Please make sure that run this command and set `epochs` = 0.

#### Opts

`method`: algorithm to quantify model, static or dynamic

`deploy_device`: pytorch now support x86 and arm, is enabled for `method` == static only

## <a id="index3">References</a>:

1. [WongKinYiu/yolov7: Implementation of paper - YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors (github.com)](https://github.com/WongKinYiu/yolov7)
2. [VainF/Torch-Pruning: [CVPR-2023] Towards Any Structural Pruning; LLaMA / CNNs / Transformers (github.com)](https://github.com/VainF/Torch-Pruning)
3. [PyTorch](https://pytorch.org/)
4. [ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5)

## <a id="index4">Contact</a>：

This repository is for [AIRS](https://airs.cuhk.edu.cn/)'s project, the author is an undergraduate student at Sun Yat sen University.

If you have any question about this repository
