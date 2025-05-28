# Machine Learning

### 项目概述

此项目为基于机器学习的对于SQL数据查询估计的实现，项目的主要实现算法为MSCN与XGBoost算法。此代码仓库为中国人民大学2025年机器学习课程的大作业。

### 关键词

基数估计、AI4DB、MSCN、XGBoost

### 项目文件结构

```
├── README.md									#本文件，对于项目的实验报告与说明
├── convert_json_to_csv.py						#能够按照指定格式，将原有json文件转化为模型需要的csv文件的程序代码
├── data										#项目数据，由于train.json过大无法上传到GitHub，已被删除
│   ├── column_min_max_vals.csv					#数据库不同页的统计信息
│   └── test_data.json							#用于测试的数据
├── environment.txt								#项目环境信息
└── learnedcardinalities						#根据参考代码修改的模型代码
    ├── data									#模型加载数据的文件夹
    │   ├── column_min_max_vals.csv				#数据库不同页的统计信息
    │   └── train.csv							#用于训练的数据
    ├── format_predictions.py					#能够按照提交格式，将输出csv文件转化为能够提交的csv格式的代码
    ├── mscn									#模型代码文件夹
    │   ├── data.py								#（原有参考代码中包含的文件）包含数据加载与训练，预测所必需的函数
    │   ├── model.py							#MSCN模型的原始实现，包含详细的模型结构
    │   ├── util.py								#（原有参考代码中包含的文件）包含程序运行所必需的辅助函数
    │   └── xgb_model.py						#XGBoost算法模型的原始实现，包含详细的模型结构
    ├── results									#模型输出预测结果的文件夹
    │   ├── predictions_test.csv				#模型原始输出的csv文件
    │   └── predictions_test_formatted.csv		#经过格式转化后，可直接用于结果提交的csv文件
    ├── train.py								#程序的主函数
    └── workloads								#模型加载预测数据的文件夹
        └── test.csv							#模型加载的预测数据
```

### 项目主要实现代码

##### MSCN模型

模型使用双层MLP对samples、predicates、joins进行编码，之后进行掩码聚合，最后连接并经过MLP与sigmoid进行预测，整体模型结构如下图：

```
           samples   predicates   joins
               │         │         │
         MLP(2层)    MLP(2层)    MLP(2层)
               ↓         ↓         ↓
          masked avg  masked avg  masked avg
               ↓         ↓         ↓
             [hid_sample | hid_pred | hid_join]
                              │
                        Concatenation
                              ↓
                        Linear + ReLU
                              ↓
                        Linear + Sigmoid
                              ↓
                            Output

```

单层MLP模型由两层线性层加上一层Relu层组成。

##### XGBoost模型

模型使用同样的编码器，不同点在于使用XGBoost结构进行回归。具体模型的推理路径如下：

```
(samples, predicates, joins) 
        ↓
  3 路 MLP + Attention
        ↓
  3 个向量拼接 (hid_sample | hid_predicate | hid_join)
        ↓
    主路径 MLP  +  残差连接
        ↓
      融合特征向量
        ↓
    → 若 is_trained = False: 输出随机 sigmoid
    → 若 is_trained = True: 用 XGBoost 回归

```

##### 对于两者模型的分析

MSCN的优点：

- 网络较深，参数较多，善于学习复杂模型分布
- 使用端到端训练，不存在模型偏差
- 掩码与平均聚合能够适应变长输入

MSCN的缺点：

- 参数量大，训练慢且依赖于大规模数据
- 网络较深，可能出现过拟合现象
- 对于优化方法，学习率等变量影响较大

XGBoost的优点：

- 特征提取快，训练速度快
- 模型相比于深层神经网络有更好的鲁棒性

XGBoost的缺点：

- 难以适应复杂的场景，在场景复杂的情况下表现较差
- 训练解耦，无法对两者模型进行同时优化

实测结果使用MSCN模型表现更好，XGBoost训练较快。

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

### 项目信息

作者：李子轩	2023202292

项目地址：[EXLYSHA/Machine-Learning: MSCN based Machine Learning homework. RUC 2025.](https://github.com/EXLYSHA/Machine-Learning)
