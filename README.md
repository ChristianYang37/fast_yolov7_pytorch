# Fast_Yolov7_Pytorch🎉️🎉️🎉️

***<center>Build Your Faster Yolov7🚀️ </center>***

## Guide

* [Installation](#index1)
* [Quick Start](#index2)
* [Papers](#index3)
* [References](#index4)

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
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml --method static
```

#### Opts

`method`: algorithm to quantify model, static or dynamic

`deploy_device`: pytorch now support x86 and arm, is enabled for `method` == static only

## <a id="index3">Papers</a>：

### 剪枝

#### （1）非结构剪枝

《DepGraph: Towards Any Structural Pruning》

Link: [2301.12900.pdf (arxiv.org)](https://arxiv.org/pdf/2301.12900.pdf)

摘要：提出了一种通用的全自动方法——依赖图(Dependency Graph, DepGraph)来显式地对层间的相互依赖进行建模，并对耦合参数进行综合分组。

Github: [WongKinYiu/yolov7: Implementation of paper - YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors (github.com)](https://github.com/WongKinYiu/yolov7)

《Network Pruning via Transformable Architecture Search》

Link: [1905.09717v5.pdf (arxiv.org)](https://arxiv.org/pdf/1905.09717v5.pdf)

摘要：对于每一层网络分化出多个同类的较小网络，通过知识蒸馏让预备网络结构学习原网络的特征图表示，选出损耗最小的网络替换原网络层。

Github: [D-X-Y/AutoDL-Projects: Automated deep learning algorithms implemented in PyTorch. (github.com)](https://github.com/D-X-Y/AutoDL-Projects)

#### （2）结构剪枝

《Structured Pruning for Deep Convolutional Neural Networks: A survey》

Link: [2303.00566v1.pdf (arxiv.org)](https://arxiv.org/pdf/2303.00566v1.pdf)

摘要：从过滤器排序方法、正则化方法、动态执行、神经结构搜索、彩票假设和剪枝的应用等方面对目前最先进的结构化剪枝技术进行了总结和比较。（综述）

Github: [he-y/Awesome-Pruning: A curated list of neural network pruning resources. (github.com)](https://github.com/he-y/Awesome-Pruning)

《Movement Pruning: Adaptive Sparsity by Fine-Tuning》

Link: [2005.07683v2.pdf (arxiv.org)](https://arxiv.org/pdf/2005.07683v2.pdf)

摘要：运动剪枝，在训练过程中保留重要性高的连接，即修建在训练过程中逐渐趋于0的连接，适合用来微调预训练模型，使参数稀疏化。

### 量化

《TRAINING WITH QUANTIZATION NOISE FOR EXTREME MODEL COMPRESSION》

Link: [2004.07320v3.pdf (arxiv.org)](https://arxiv.org/pdf/2004.07320v3.pdf)

摘要：通过在训练中引入随机量化、部分量化增强模型对精度损失的鲁棒性。

《LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale》

Link: [2208.07339v2.pdf (arxiv.org)](https://arxiv.org/pdf/2208.07339v2.pdf)

摘要：对矩阵乘法中的每个内积使用独立的归一化常数的矢量量化，以量化大多数特征，通过对列和行规范化常数的外积进行反规范化处理来恢复矩阵乘法的输出，效果近乎无损。

## <a id="index4">References</a>:

1. [WongKinYiu/yolov7: Implementation of paper - YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors (github.com)](https://github.com/WongKinYiu/yolov7)
2. [VainF/Torch-Pruning: [CVPR-2023] Towards Any Structural Pruning; LLaMA / CNNs / Transformers (github.com)](https://github.com/VainF/Torch-Pruning)
3. [PyTorch](https://pytorch.org/)
