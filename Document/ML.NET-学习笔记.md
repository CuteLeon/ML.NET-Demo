# ML.NET-学习笔记

> Link: https://docs.microsoft.com/zh-cn/dotnet/machine-learning/how-does-mldotnet-work
>
> Samples: https://github.com/dotnet/machinelearning-samples
>
> Tutorials: https://github.com/dotnet/samples/tree/master/machine-learning/tutorials
>
> ML.NETModelBuilder: https://marketplace.visualstudio.com/items?itemName=MLNET.07
>
> ML.NETModelBuilder-Tutorial: https://dotnet.microsoft.com/learn/machinelearning-ai/ml-dotnet-get-started-tutorial/intro
>
> Author: Leon
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

​	在实际应用程序中，模型训练和评估代码将与预测分离。 

​	事实上，这两项活动通常由单独的团队执行。 模型开发团队可以保存模型以便用于预测应用程序。



# 教程

## 分析情绪(二元分类)



## 对支持问题进行分类(多类分类)



## 预测价格(回归)



## 对鸢尾花进行分类(K 平均值聚类分析)



## 影片推荐系统(矩阵因子分解)



# 加载数据

## 使用列特性注释数据模型

### LoadColumn

​	指定属性的列索引，只有从文件加载的数据才需要此特性。

**用法：**

- [LoadColumn(int index)]
  - 指定第 index 列数据；
- [LoadColumn(int start, int end)]
  - 指定第 {start} 至 {end} 列数据；
- [LoadColumn(int[] indexs)]
  - 指定数组 indexs 对应的所有列数据；

### VectorType

​	指定多列数据读取为向量格式，需要此多列数据的类型相同。

**用法：**

- [VectorType()]
  - 将成员标记为具有未知大小的一维数组；
- [VectorType(int size)]
  - 将成员标记为 {size} 指定大小的一维数组；
  - size = 0 表示矢量类型的长度未知；
- [VectorType(int[] dimensions)]
  - 将成员标记为 {dimensions} 指定维度的多维数组；
  - 维度数组内应为非负数；
  - dimensions = 0 表示多维数组在此维度的长度未知；

### ColumnName

​	指定列名称更改为该属性名称以外的其他名称。

​	在内存中创建对象时，仍然使用该属性名称创建对象；但是对于数据处理和生成机器学习模型，ML.NET 使用 此特性中提供的值覆盖并引用该属性。



```csharp
public class HousingData
{
    [LoadColumn(0)]
    public float Size { get; set; }
 
    [LoadColumn(1, 3)]
    [VectorType(3)]
    public float[] HistoricalPrices { get; set; }

    [LoadColumn(4)]
    [ColumnName("Label")]
    public float CurrentPrice { get; set; }
}
```



## 从单个文件加载数据

​	若要从文件加载数据，请使用 `LoadFromTextFile` 方法以及要加载的数据的数据模型。 

​	由于 `separatorChar` 参数**默认为制表符分隔**，因此请根据需要为数据文件更改该参数。 

​	如果文件有标头，请将 `hasHeader` 参数设置为 `true`，以忽略文件中的第一行并开始从第二行加载数据。

```csharp
//Create MLContext
MLContext mlContext = new MLContext();

//Load Data
IDataView data = mlContext.Data.LoadFromTextFile<HousingData>("my-data-file.csv", separatorChar: ',', hasHeader: true);
```



## 从多个目录中的文件加载

​	若要从多个目录加载数据，请使用 `CreateTextLoader` 方法创建 `TextLoader`。 然后，使用 `TextLoader.Load` 方法并指定单个文件路径（不能使用通配符）。

```csharp
//Create MLContext
MLContext mlContext = new MLContext();

// Create TextLoader
TextLoader textLoader = mlContext.Data.CreateTextLoader<HousingData>(separatorChar: ',', hasHeader: true);

// Load Data
IDataView data = textLoader.Load("DataFolder/SubFolder1/1.txt", "DataFolder/SubFolder2/1.txt");
```

## 从流式处理源加载数据

​	除了加载存储在磁盘上的数据外，ML.NET 还支持从各种流式处理源加载数据，这些源包括但不限于：

- 内存中集合
- JSON/XML
- 数据库

> 请注意，在使用流式处理源时，ML.NET 预计输入采用内存中集合的形式。 因此，在使用 JSON/XML 等源时，请确保将数据格式化为内存中集合。

```csharp
IDataView dataView = mlContext.Data.LoadFromEnumerable<Model>(this.GetModels());

IEnumerable<Model> GetModels()
	=> System.Linq.Enumerable.Range(0, 100).Select(index => new Model(index));
```



# 准备数据

## 筛选数据