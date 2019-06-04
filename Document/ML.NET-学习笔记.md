# ML.NET-学习笔记

> Link: https://docs.microsoft.com/zh-cn/dotnet/machine-learning/how-does-mldotnet-work
>
> Samples: https://github.com/dotnet/machinelearning-samples
>
> Tutorials: https://github.com/dotnet/samples/tree/master/machine-learning/tutorials
>
> API: https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.data?view=ml-dotnet
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

​	`DataOperationsCatalog` 包含一组筛选操作，这些操作接收包含所有数据的 `IDataView`，并返回仅包含关注数据点的 `IDataView`。 

```csharp
IDataView filteredData = mlContext.Data.FilterRowsByColumn(dataView, "Price", lowerBound: 200000, upperBound: 1000000);
```

## 替换缺失值

​	处理缺失值的一种方法是使用给定类型的默认值（如有）或其他有意义的值（例如数据中的平均值）替换它们。

> ReplaceMissingValues 仅适用于数字类型

```csharp
using Microsoft.ML.Transforms;

// 训练数据集合
IDataView trainingDataView = ...;
// Mean：使用平均值填充缺失的值
var replacementEstimator = mlContext.Transforms.ReplaceMissingValues(
    "Price", 
    replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean);
// 训练评估器
ITransformer replacementTransformer = replacementEstimator.Fit(trainingDataView);
// 返回填充完成的数据集
trainingDataView = replacementTransformer.Transform(trainingDataView);
```
## 使用规范化程序

### 最大-最小规范化

​	规范化是一种数据预处理技术，用于标准化比例不同的特征，这有助于算法更快地融合。

```csharp
IDataView trainingDataView = ...;
var minMaxEstimator = mlContext.Transforms.NormalizeMinMax("Price");
ITransformer minMaxTransformer = minMaxEstimator.Fit(trainingDataView);
IDataView transformedData = minMaxTransformer.Transform(trainingDataView);
```

​	将一组任何区间的数字转换为一组在0~1区间内的数字，值为 当前数字与数组内最大值之比。

​	例如：原始价格值 `[200000,100000]` 使用 `MinMax` 规范化公式转换为 `[ 1, 0.5 ]`。

### 分箱

​	分箱将连续值转换为输入的离散表示形式。

```csharp
IDataView trainingDataView = ...;
var binningEstimator = mlContext.Transforms.NormalizeBinning("Price", maximumBinCount: 2);
var binningTransformer = binningEstimator.Fit(trainingDataView);
IDataView transformedData = binningTransformer.Transform(trainingDataView);
```

​	 `maximumBinCount` 参数使你可以指定对数据进行分类所需的箱数；

## 使用分类数据

​	在用于生成机器学习模型之前，需要将非数字分类数据转换为数字。

```csharp
IDataView trainingDataView = ...;
var categoricalEstimator = mlContext.Transforms.Categorical.OneHotEncoding("VehicleType");
ITransformer categoricalTransformer = categoricalEstimator.Fit(trainingDataView);
IDataView transformedData = categoricalTransformer.Transform(trainingDataView);
```

## 使用文本数据

​	将一系列转换应用于输入文本列，从而生成表示 lp 规范化字词和 n 元语法的数字向量。将复杂的文本处理步骤合并到一个 `EstimatorChain` 中以消除干扰，并可能根据需要减少所需的处理资源量。

```csharp
IDataView trainingDataView = ...;
var textEstimator = mlContext.Transforms.Text.FeaturizeText("Description");
ITransformer textTransformer = textEstimator.Fit(trainingDataView);
IDataView transformedData = textTransformer.Transform(trainingDataView);
```

**原始文本：This is a good product**

| Transform              | 说明                               | 结果                                            |
| :--------------------- | :--------------------------------- | :---------------------------------------------- |
| NormalizeText          | 默认情况下将所有字母转换为小写字母 | this is a good product                          |
| TokenizeWords          | 将字符串拆分为单独的字词           | ["this","is","a","good","product"]              |
| RemoveDefaultStopWords | 删除 *is* 和 *a* 等非索引字        | ["good","product"]                              |
| MapValueToKey          | 根据输入数据将值映射到键（类别）   | [1,2]                                           |
| ProduceNGrams          | 将文本转换为连续单词的序列         | [1,1,1,0,0]                                     |
| NormalizeLpNorm        | 按缩放的 lp 规范缩放输入           | [ 0.577350529, 0.577350529, 0.577350529, 0, 0 ] |



# 训练和评估模型

## 拆分数据用于训练和测试

​	机器学习模型旨在识别训练数据中的模式。 这些模式用于使用新数据进行预测。

​	使用 `TrainTestSplit` 方法将数据拆分为训练集和测试集。 结果将是一个 `TrainTestData` 对象，其中包含两个 `IDataView` 成员，一个用于训练集，另一个用于测试集。 数据拆分百分比由 `testFraction` 参数确定。 

```csharp
// 从训练数据集分拆 20% 作为测试数据集
DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
IDataView trainData = dataSplit.TrainSet;
IDataView testData = dataSplit.TestSet;
```

## 准备数据

​	在训练机器学习模型之前，需要对数据进行预处理。

> ML.NET 算法对输入列类型存在约束。 此外，如果未指定任何值，则默认值会用于输入和输出列名。

### 使用预期的列类型

​	ML.NET 中的机器学习算法预期使用大小已知的浮点向量作为输入。 

​	当所有数据都已经是数字格式并且打算一起处理（即图像像素）时，将 `VectorType` 属性应用于数据模型。

​	如果数据不全为数字格式，并且想要单独对每个列应用不同的数据转换，请在处理所有列后使用 `Concatenate` 方法，以将所有单独的列合并为一个特征向量并将特征向量输出到新列。

```csharp
IEstimator<ITransformer> dataPrepEstimator =
    // 将 Size 和 HistoricalPrices 列合并为 Features 向量
    mlContext.Transforms.Concatenate("Features", "Size", "HistoricalPrices")
    	// 使用最大-最小规范化 Features 列
        .Append(mlContext.Transforms.NormalizeMinMax("Features"));
```

### 使用默认列名

​	未指定列名时，ML.NET 算法会使用默认列名。 

​	所有训练程序都有一个名为 `featureColumnName`的参数可用于算法的输入，并且在适用情况下，它们还有一个用于预期值的名为 `labelColumnName`的参数。 默认情况下，这些值分别为 `'Features'` 和 `'Label'`。

​	也可以自定义指定 label 和 feature 列名称：

```csharp
var UserDefinedColumnSdcaEstimator = mlContext.Regression.Trainers.Sdca(labelColumnName: "MyLabelColumnName", featureColumnName: "MyFeatureColumnName");
```

## 训练机器学习模型

​	对数据进行预处理后，使用 [`Fit`](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.trainerestimatorbase-2.fit) 方法通过 [`StochasticDualCoordinateAscent`](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcaregressiontrainer) 回归算法训练机器学习模型。

```csharp
// 定义估算器
var sdcaEstimator = mlContext.Regression.Trainers.Sdca();

// 训练机器学习模型
var trainedModel = sdcaEstimator.Fit(transformedTrainingData);
```

## 提取模型参数

​	训练模型后，提取已学习的 [`ModelParameters`](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.modelparametersbase-1) 用于检查或重新训练，提供经过训练的模型的偏差和已学习的系数或权重。

```csharp
var trainedModelParameters = trainedModel.Model as LinearRegressionModelParameters;
```

> 不同的机器学习任务可转换为不同的 ModelParameters 类型提供神经元的权重和偏置：Microsoft.ML.Trainers.*Parameters

## 评估模型质量

​	若要帮助选择性能最佳的模型，必须评估其在测试数据中的性能。

> 不同的机器学习任务生成不同类型的评估指标对象：Microsoft.ML.Data.*Metrics



# 使用交叉验证训练和评估机器学习模型

​	交叉验证是一种训练和模型评估技术，**可将数据拆分为多个分区，并利用这些分区训练多个算法**。此技术通过保留来自训练过程的数据来提高模型的可靠性。 除提高不可见观测的性能之外，在数据受限的环境中，它还可用作使用较小数据集训练模型的有效工具。

## 数据和数据模型

```
Size (Sq. ft.), HistoricalPrice1 ($), HistoricalPrice2 ($), HistoricalPrice3 ($), Current Price ($)
620.00, 148330.32, 140913.81, 136686.39, 146105.37
550.00, 557033.46, 529181.78, 513306.33, 548677.95
1127.00, 479320.99, 455354.94, 441694.30, 472131.18
1120.00, 47504.98, 45129.73, 43775.84, 46792.41
```

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

## 准备数据

```csharp
IEstimator<ITransformer> dataPrepEstimator = 
    mlContext.Transforms.Concatenate("Features", new string[] { "Size", "HistoricalPrices" })
        .Append(mlContext.Transforms.NormalizeMinMax("Features"));
ITransformer dataPrepTransformer = dataPrepEstimator.Fit(data);
IDataView transformedData = dataPrepTransformer.Transform(data);
```

## 使用交叉验证训练模型

```csharp
IEstimator<ITransformer> sdcaEstimator = mlContext.Regression.Trainers.Sdca();
// numberOfFolds：交叉验证层数
var cvResults = mlContext.Regression.CrossValidate(transformedData, sdcaEstimator, numberOfFolds: 5);
```

`CrossValidate` 执行以下操作：

1. 将数据分为多个分区，数量为 `numberOfFolds` 参数中指定的值。 
2. 使用训练数据集上的指定机器学习算法估算器在每个分区上训练模型。
3. 每个模型的性能在测试数据集上使用 `Evaluate` 方法进行评估。
4. 为每个模型返回模型及其指标(CrossValidationResult类型的集合)

## 提取指标

```csharp
IEnumerable<double> rSquared = cvResults.Select(fold => fold.Metrics.RSquared);
```

## 选择性能最好的模型

​	使用 R 平方等指标按性能最好到性能最差的顺序选择模型。 然后，选择性能最好的模型来进行预测或执行其他操作。

```csharp
var preferModel = cvResults
    .OrderByDescending(result => result.Metrics.RSquared)
    .FirstOrDefault()?
    .Model;
```



