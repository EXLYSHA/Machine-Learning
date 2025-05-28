# Machine Learning

### 项目概述

此项目为基于机器学习的对于SQL数据查询估计的实现，项目的主要实现算法为MSCN与XGBoost算法。此代码仓库为中国人民大学2025年机器学习课程的大作业。

### 关键词

基数估计、AI4DB、MSCN、XGBoost

### 项目文件结构

### 项目主要实现代码

### 代码使用

环境配置

```
conda install --file environment.txt
```

快速训练

```
cd learnedcardinalities
python train.py test
```

参数说明

```
testset 	#workload的名称，使用workload的文件名即可
--queries 	#训练使用的查询数量，最大值为60000，超过60000的值一律按照60000处理，默认为10000
--epochs	#训练的epoch数量，默认为10
--batch		#batch大小，默认为1024
--hid		#特征提取的隐藏层大小，默认为256
--cuda		#是否强制使用CUDA，默认为模型自动选择
--use-xgb	#是否使用XGBoost模型，未指定此参数将使用MSCN模型训练
```

### 优化方向

- 使用更加复杂的模型结构，如引入自回归层对MSCN模型进行优化
- 尝试使用其他树形结构，如使用LightGBM替换XGBoost树
- 使用文本embedding替换one-hot编码，避免one-hot带来的因为特征工程不足的精度损失
- 使用更加复杂的特征，如查询中更多的信息或更多的特征工程
- 使用查询采样、查询数据等复杂的多模态数据，优化MSCN模型或使用DeepDB等模型
- 使用深层神经网络并使用残差连接，避免模型的过拟合问题

### 参考文献

[1] Andreas Kipf, Thomas Kipf, Bernhard Radke, Viktor Leis, Peter A. Boncz, Alfons Kemper: Learned Cardinalities: Estimating Correlated Joins with Deep Learning. CIDR 2019.

[2] Benjamin Hilprecht, Andreas Schmidt, Moritz Kulessa, Alejandro Molina, Kristian Kersting, Carsten Binnig: DeepDB: Learn from Data, not from Queries! VLDB 2020.

### 作者

李子轩	2023202292
