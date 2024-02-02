# 数据集的统计分析


## 数据的输入：
1. 转为id的三元组
2. 未转为id的三元组 + id字典
3. 未转id的三元组
4. 加入reverse 三元组

# 数据集的构建
1. 输出：正负样本的构建
    * 负采样：
    * 按照权重的采样

2. 输出：ground-truth 的构建： 目前是两个数据集，建议放到一起
    * 1-1的构建
    * 1-N的构建

# 损失函数
* 正负样本的损失函数
* ground-truth 和 prediction 之间的损失函数

# 针对数据集和损失的不同构建两个训练和测试的方式 

# 指标问题
新增一些测试指标和新的测试方法


# 正则化问题

# 可视化分析工具



# 模型实现：
    支持5中计算模式：主要是为了提升一点计算速度
    模型不同初始化的方式
    模型之间的参数形式

应用于科研领域：让代码尽量的清晰易懂，容易进行扩充
方法之间尽量独立，便于扩充和自己组合，然后提供一个顶层一点的组合


1. 负采样的方法：
    如果基于封闭世界假设，则可以全部进行负采样
    如果基于随机封闭世界假设，则是随机方法选择负样本
    如果基于RotatE的采样，需要增加权重

2. 数据集的形式：
数据会有两个大类：
第一类是h,r,t 全部都有， 第二类是只有hr或rt，表示h或t为全部实体，这里不构建数据集能够有效减小显存占用。
无论何种数据集，模型只需要计算对应的


3. 模型的形式
    模型内自定义参数


4. 损失定义

5. 正则方法

6. 超参选择方法

日志的配置

首先：
实现最常用的MRL：
$$
L(\Delta) = \max(0, \lambda + \Delta)
$$
$\Delta$ is equal to the result of negative samples score sub positive samples score.

