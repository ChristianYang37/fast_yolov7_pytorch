# Papers：

## 剪枝

### （1）非结构剪枝

《DepGraph: Towards Any Structural Pruning》

Link: [2301.12900.pdf (arxiv.org)](https://arxiv.org/pdf/2301.12900.pdf)

摘要：提出了一种通用的全自动方法——依赖图(Dependency Graph, DepGraph)来显式地对层间的相互依赖进行建模，并对耦合参数进行综合分组。

Github: [WongKinYiu/yolov7: Implementation of paper - YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors (github.com)](https://github.com/WongKinYiu/yolov7)

《Network Pruning via Transformable Architecture Search》

Link: [1905.09717v5.pdf (arxiv.org)](https://arxiv.org/pdf/1905.09717v5.pdf)

摘要：对于每一层网络分化出多个同类的较小网络，通过知识蒸馏让预备网络结构学习原网络的特征图表示，选出损耗最小的网络替换原网络层。

Github: [D-X-Y/AutoDL-Projects: Automated deep learning algorithms implemented in PyTorch. (github.com)](https://github.com/D-X-Y/AutoDL-Projects)

### （2）结构剪枝

《Structured Pruning for Deep Convolutional Neural Networks: A survey》

Link: [2303.00566v1.pdf (arxiv.org)](https://arxiv.org/pdf/2303.00566v1.pdf)

摘要：从过滤器排序方法、正则化方法、动态执行、神经结构搜索、彩票假设和剪枝的应用等方面对目前最先进的结构化剪枝技术进行了总结和比较。（综述）

Github: [he-y/Awesome-Pruning: A curated list of neural network pruning resources. (github.com)](https://github.com/he-y/Awesome-Pruning)

《Movement Pruning: Adaptive Sparsity by Fine-Tuning》

Link: [2005.07683v2.pdf (arxiv.org)](https://arxiv.org/pdf/2005.07683v2.pdf)

摘要：运动剪枝，在训练过程中保留重要性高的连接，即修建在训练过程中逐渐趋于0的连接，适合用来微调预训练模型，使参数稀疏化。

## 量化

《TRAINING WITH QUANTIZATION NOISE FOR EXTREME MODEL COMPRESSION》

Link: [2004.07320v3.pdf (arxiv.org)](https://arxiv.org/pdf/2004.07320v3.pdf)

摘要：通过在训练中引入随机量化、部分量化增强模型对精度损失的鲁棒性。

《LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale》

Link: [2208.07339v2.pdf (arxiv.org)](https://arxiv.org/pdf/2208.07339v2.pdf)

摘要：对矩阵乘法中的每个内积使用独立的归一化常数的矢量量化，以量化大多数特征，通过对列和行规范化常数的外积进行反规范化处理来恢复矩阵乘法的输出，效果近乎无损。