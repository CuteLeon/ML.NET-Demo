# ML.NET-学习笔记

> Link: https://docs.microsoft.com/zh-cn/dotnet/machine-learning/how-does-mldotnet-work
>
> author: Leon
>
> Date: 2019/05/24

> ML.NET API 有两组包：发布组件和预览组件。 发布 API 包含用于数据处理的组件、用于二元分类、多类分类、回归、异常情况检测和排名的算法以及模型保存和加载等等！ 预览 API 包含 ONNX 和 TensorFlow 模型处理、推荐任务算法以及用于处理时序数据的组件。

# 概述

## 什么是 ML.NET 以及它如何工作？

​	ML.NET 使你能够将机器学习添加到 .NET 应用程序中。 借助此功能，可以使用应用程序的可用数据进行自动预测。

​	可以使用 ML.NET 进行的预测类型的示例包括：

| 类型            | 示例                                                   |
| :-------------- | :----------------------------------------------------- |
| 分类/类别划分   | 自动将客户反馈划分为正面和负面类别                     |
| 回归/预测连续值 | 根据大小和位置预测房屋价格                             |
| 异常情况检测    | 检测欺诈性银行交易                                     |
| 建议            | 根据在线购物者之前的购买情况向其建议可能想要购买的产品 |

## 代码工作流

以下关系表示应用程序代码结构，以及模型开发的迭代过程：

- 将训练数据收集并加载到 **IDataView** 对象中
- 指定操作的管道，以提取特征并应用机器学习算法
- 通过在管道上调用 **Fit()** 来训练模型
- 评估模型并通过迭代进行改进
- 将模型保存为二进制格式，以便在应用程序中使用
- 将模型加载回 **ITransformer** 对象
- 通过调用 **CreatePredictionEngine.Predict()** 进行预测

![](Images\mldotnet-annotated-workflow.png)

## 机器学习模型

​	用于查找模型参数的数据称为**训练数据**。 

​	机器学习模型的输入称为**特征**。

​	用于训练机器学习模型的真值称为**标签**。

### 更复杂

​	更复杂的模型使用事务文本描述将事务分类为类别。

## 数据准备

​	在大多数情况下，可用的数据不适合直接用于训练机器学习模型。 需要准备或预处理原始数据，然后才能将其用于查找模型的参数。 数据可能需要从字符串值转换为数字表示形式。 输入数据中可能会包含冗余信息。 可能需要缩小或放大输入数据的维度。 数据可能需要进行规范化或缩放。

## 模型评估

​	每种类型的机器学习任务都具有用于根据测试数据集评估模型的准确性和精确性的指标。

​	需要进行更多调整才能获得良好的模型指标。

## ML.NET 体系结构

​	ML.NET 应用程序从 [MLContext](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.mlcontext) 对象开始。 此单一实例对象包含**目录**。 目录是用于数据加载和保存、转换、训练程序和模型操作组件的工厂。 每个目录对象都具有创建不同类型的组件的方法：

|                |              |                                                              |
| :------------- | :----------- | :----------------------------------------------------------- |
| 数据加载和保存 |              | [DataOperationsCatalog](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.dataoperationscatalog) |
| 数据准备       |              | [TransformsCatalog](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.transformscatalog) |
| 训练算法       | 二元分类     | [BinaryClassificationCatalog](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.binaryclassificationcatalog) |
|                | 多类分类     | [MulticlassClassificationCatalog](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.multiclassclassificationcatalog) |
|                | 异常情况检测 | [AnomalyDetectionCatalog](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.anomalydetectioncatalog) |
|                | 排名         | [RankingCatalog](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.rankingcatalog) |
|                | 回归测试     | [RegressionCatalog](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.regressioncatalog) |
|                | 建议         | [RecommendationCatalog](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.recommendationcatalog) |
| 模型使用       |              | [ModelOperationsCatalog](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.modeloperationscatalog) |

### 生成管道

​	每个目录中都有一组扩展方法。

### 定型模型

​	调用 `Fit()` 使用输入训练数据来估算模型的参数。 这称为训练模型。

​	上述线性回归模型有两个模型参数：**偏差**和**权重**。

### 使用模型

​	可以将输入数据批量转换为预测，也可以一次转换一个输入。

### 数据模型和架构

​	ML.NET 机器学习管道的核心是 [DataView](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.idataview) 对象。

​	管道中的每个转换都有一个输入架构（转换期望在其输入中看到的数据名称、类型和大小）；以及一个输出架构（转换在转换后生成的数据名称、类型和大小）。

​	如果管道中一个转换的输出架构与下一个转换的输入架构不匹配，ML.NET 将引发异常。

​	数据视图对象具有列和行。 每个列都有名称、类型和长度。

​	**DataView 对象的一个重要属性是它们被惰性求值。** 数据视图仅在模型训练和评估以及数据预测期间加载及运行。 在编写和测试 ML.NET 应用程序时，可以使用 Visual Studio 调试程序通过调用 Preview 方法来浏览任何数据视图对象。

> 不要在生产代码中使用 Preview 方法，因为它会大幅降低性能。

### 模型部署

