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
> MSDN: https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml?view=ml-dotnet
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
Helper.PrintLine("加载训练数据集...");
IDataView sourceDataView = mlContext.Data.LoadFromTextFile<HousingData>(
    TrainDataPath,
    separatorChar: ',',
    hasHeader: true,
    trimWhitespace: true);

Helper.PrintLine("创建数据初始化对象...");
IEstimator<ITransformer> dataPrepEstimator =
    mlContext.Transforms.Concatenate("Features", new string[] { "Size", "HistoricalPrices" })
        .Append(mlContext.Transforms.NormalizeMinMax("Features"));

Helper.PrintLine("初始化数据...");
ITransformer dataPrepTransformer = dataPrepEstimator.Fit(sourceDataView);
IDataView transformedData = dataPrepTransformer.Transform(sourceDataView);
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
Helper.PrintLine($"输出模型性能：\n\t{string.Join("\n\t", cvResults.Select(result => $">>> Fold: {result.Fold} —————— >>>\n\tRSquared: {result.Metrics.RSquared}\n\tLossFunction: {result.Metrics.LossFunction}\n\tMeanAbsoluteError: {result.Metrics.MeanAbsoluteError}\n\tMeanSquaredError: {result.Metrics.MeanSquaredError}\n\tRootMeanSquaredError: {result.Metrics.RootMeanSquaredError}"))}");
```

## 选择性能最好的模型

​	使用 R 平方等指标按性能最好到性能最差的顺序选择模型。 然后，选择性能最好的模型来进行预测或执行其他操作。

```csharp
var preferModel = cvResults
    .OrderByDescending(result => result.Metrics.RSquared)
    .FirstOrDefault()?
    .Model;
```

# 在处理期间检查中间数据值

​	在 ML.NET 的加载、处理和训练步骤中检查值。

## 将 IDataView 转换为 IEnumerable

​	检查 `IDataView` 的值的最快方法之一是将其转换为 `IEnumerable`。 若要将 `IDataView` 转换为 `IEnumerable`，请使用 `CreateEnumerable` 方法。

​	若要优化性能，请将 `reuseRowObject` 的值设置为 `true`。 如果这样做，将在评估当前行的数据时延迟填充相同的对象，而不是为数据集中的每一行创建一个新对象。

```csharp
System.Collections.Generic.IEnumerable<Model> rows =
	mlContext.Data.CreateEnumerable<Model>(trainingDataView, reuseRowObject: true);

foreach (var row in rows)
{
}
```

​	如果只需要访问部分数据或特定索引，请使用 `CreateEnumerable` 并将 `reuseRowObject` 参数值设置为 `false`，以便为数据集中每个请求的行创建一个新对象。 然后，将 `IEnumerable` 转换为数组或列表。

## 检查单个列中的值

​	在模型生成过程中的任何时候，都可以使用 `GetColumn` 方法访问 `IDataView` 的单个列中的值。`GetColumn` 方法将单个列中的所有值都返回为 `IEnumerable`。

```csharp
IEnumerable<float> sizeColumn = data.GetColumn<float>("Size").ToList();
```

## 一次检查一行 IDataView 值

​	`IDataView` 延迟求值。 若要循环访问 `IDataView` 的各行，而不按本文档前面部分所示转换为 `IEnumerable`，请通过使用 `GetRowCursor` 方法并传入 `IDataView` 的 DataViewSchema 作为参数来创建 `DataViewRowCursor`。 然后，若要循环访问各行，请使用 `MoveNext` 游标方法以及 `ValueGetter` 委托从每个列中提取相应的值。

> 出于性能考虑，ML.NET 中的向量使用 `VBuffer` 而不是本机集合类型

```csharp
using Microsoft.ML.Data;

// 获取数据集大纲
DataViewSchema schema = trainingDataView.Schema;
// 获取数据集游标
using (DataViewRowCursor cursor = trainingDataView.GetRowCursor(schema))
{
    float size = default;
    Microsoft.ML.Data.VBuffer<float> historicalPrices = default;
    float currentPrice = default;

    // 定义读取数据的委托
    ValueGetter<float> sizeDelegate = cursor.GetGetter<float>(schema[0]);
    ValueGetter<Microsoft.ML.Data.VBuffer<float>> historicalPriceDelegate = cursor.GetGetter<Microsoft.ML.Data.VBuffer<float>>(schema[1]);
    ValueGetter<float> currentPriceDelegate = cursor.GetGetter<float>(schema[2]);

    // 使用迭代器遍历数据集，并使用委托读取数据
    while (cursor.MoveNext())
    {
        sizeDelegate.Invoke(ref size);
        historicalPriceDelegate.Invoke(ref historicalPrices);
        currentPriceDelegate.Invoke(ref currentPrice);
    }
}
```

## 预览数据子集的预处理或训练结果

> **请勿在生产代码中使用 `Preview`，因为它专用于调试，可能会降低性能**

​	模型生成过程是实验性的和迭代的。 若要预览对数据子集预处理或训练机器学习模型后的数据，请使用可返回 `DataDebuggerPreview` 的 `Preview` 方法。 其结果为一个具有 `ColumnView` 和 `RowView`属性的对象，这两个属性都是 `IEnumerable` 并包含特定列或行中的值。 使用 `maxRows` 参数指定要应用转换的行数。



# 使用排列特征重要性解释模型预测

​	借助排列特征重要性 (PFI) 理解特征对预测的贡献，了解如何解释 ML.NET 机器学习模型预测。

​	机器学习模型通常被视为黑盒，它们接收输入并生成输出。 人们对影响输出的中间步骤或特征之间的交互了解甚少。随着机器学习被引入日常生活的更多方面，理解机器学习模型为何做出其决策变得至关重要。模型的可解释性水平越高，就越有信心接受或拒绝模型做出的决策。

​	有各种技术被用于解释模型，其中之一是 PFI。 PFI 是一种用于解释分类和回归模型的技术。其**工作原理是一次随机为整个数据集随机抽取数据的一个特征，并计算关注性能指标的下降程度。 变化越大，特征就越重要。通过突出显示最重要的特征，模型生成器可以专注于使用一组更有意义的特征，这可能会减少干扰和训练时间**。

## 加载数据

```
1,24,13,1,0.59,3,96,11,23,608,14,13,32
4,80,18,1,0.37,5,14,7,4,346,19,13,41
2,98,16,1,0.25,10,5,1,8,689,13,36,12
```

```csharp
class HousingPriceData
{
    [LoadColumn(0)]
    public float CrimeRate { get; set; }

    [LoadColumn(1)]
    public float ResidentialZones { get; set; }

    [LoadColumn(2)]
    public float CommercialZones { get; set; }

    [LoadColumn(3)]
    public float NearWater { get; set; }

    [LoadColumn(4)]
    public float ToxicWasteLevels { get; set; }

    [LoadColumn(5)]
    public float AverageRoomNumber { get; set; }

    [LoadColumn(6)]
    public float HomeAge { get; set; }

    [LoadColumn(7)]
    public float BusinessCenterDistance { get; set; }

    [LoadColumn(8)]
    public float HighwayAccess { get; set; }

    [LoadColumn(9)]
    public float TaxRate { get; set; }

    [LoadColumn(10)]
    public float StudentTeacherRatio { get; set; }

    [LoadColumn(11)]
    public float PercentPopulationBelowPoverty { get; set; }

    [LoadColumn(12)]
    [ColumnName("Label")]
    public float Price { get; set; }
}
```

## 定型模型

```csharp
Helper.PrintLine($"使用 PFI 解释模型...");

MLContext mlContext = new MLContext();

Helper.PrintLine("加载训练数据集...");
IDataView trainDataView = mlContext.Data.LoadFromTextFile<HousingPriceData>(TrainDataPath, separatorChar: ',');

Helper.PrintLine("获取特征成员名称...");
string[] featureColumnNames = trainDataView.Schema
    .Select(column => column.Name)
    .Where(columnName => columnName != "Label")
    .ToArray();

Helper.PrintLine("创建数据初始化对象...");
IEstimator<ITransformer> dataPrepEstimator = mlContext.Transforms.Concatenate("Features", featureColumnNames)
    .Append(mlContext.Transforms.NormalizeMinMax("Features"));

Helper.PrintLine("初始化数据...");
ITransformer dataPrepTransformer = dataPrepEstimator.Fit(trainDataView);
IDataView preprocessedTrainData = dataPrepTransformer.Transform(trainDataView);

Helper.PrintLine("创建数据估算器对象...");
SdcaRegressionTrainer sdcaEstimator = mlContext.Regression.Trainers.Sdca();

Helper.PrintSplit();
Helper.PrintLine($"开始训练神经网络...");
var sdcaModel = sdcaEstimator.Fit(preprocessedTrainData);
Helper.PrintLine($"训练神经网络完成");
Helper.PrintSplit();
```

## 使用排列特征重要性 (PFI) 解释模型

```csharp
ImmutableArray<RegressionMetricsStatistics> pfi = mlContext.Regression.PermutationFeatureImportance(
    sdcaModel,
    preprocessedTrainData,
    permutationCount: 3);
```

​	在训练数据集上使用 `PermutationFeatureImportance` 的结果是 `RegressionMetricsStatistics` 对象的 `ImmutableArray`。 `RegressionMetricsStatistics` 提供 `RegressionMetrics` 的多个观测值的均值和标准差等摘要统计信息，观测值数量等于 `permutationCount` 参数指定的排列数。

​	重要性(R 平方指标的绝对平均下降)可随后按从最重要到最不重要的顺序排序。

```csharp
Helper.PrintLine("按相关性排序特征...");
var featureImportanceMetrics = pfi
    .Select((metric, index) => new { index, metric.RSquared })
    .OrderByDescending(myFeatures => Math.Abs(myFeatures.RSquared.Mean))
    .ToArray();
Helper.PrintLine($"特征 PFI:\n\t{string.Join("\n\t", featureImportanceMetrics.Select(feature => $">>> {featureColumnNames[feature.index]}\n\tMean: {feature.RSquared.Mean:F6}\n\tStandardDeviation: {feature.RSquared.StandardDeviation:F6}\n\tStandardError: {feature.RSquared.StandardError:F6}"))}");
```



# 保存和加载经过训练的模型

​	在整个模型生成过程中，模型位于内存中，并且可以在整个应用程序生命周期中访问。 但是，一旦应用程序停止运行，而模型未在本地或远程的某个位置保存，则无法再访问该模型。 通常情况下，在其他应用程序中训练模型之后，某些时候会使用模型进行推理或重新训练。 因此，存储模型很重要。

​	由于大部分模型和数据准备管道都继承自同一组类，这些组件的保存和加载方法签名相同。 根据用例，可以将数据准备管道和模型合并为单个 `EstimatorChain`（输出单个 `ITransformer`），也可将它们分隔，从而为其各自创建单独的 `ITransformer`。

## 在本地保存模型

​	保存模型时，需要以下两项：

1. 模型的 `ITransformer`。
2. `ITransformer` 预期输入的 `DataViewSchema`。

​	训练模型后，通过 `Save` 方法使用输入数据的 `DataViewSchema` 将经过训练的模型保存到名为 `model.zip` 的文件中。

```csharp
mlContext.Model.Save(trainedModel, data.Schema, "model.zip");
```

## 加载本地存储的模型

​	在单独的应用程序或进程中，配合使用 `Load` 方法和文件路径将经过训练的模型载入应用程序。

```csharp
ITransformer trainedModel = mlContext.Model.Load("model.zip", out DataViewSchema modelSchema);
```

## 加载远程存储的模型

​	若要将存储在远程位置的数据准备管道和模型加载到应用程序中，请使用 `Stream`，而不要使用 `Load` 方法中的文件路径。

```csharp
using System.Net.Http;

MLContext mlContext = new MLContext();

ITransformer trainedModel;
using (HttpClient client = new HttpClient())
{
	Stream modelFile = await client.GetStreamAsync("{远程文件服务地址}");
	trainedModel = mlContext.Model.Load(modelFile, out DataViewSchema modelSchema);
}
```

## 使用单独的数据准备和模型管道

> 使用单独的数据准备和模型训练管道是可选方案。 管道的分离使得用户能够更轻松地检查已学习的模型参数。 对于预测，保存和加载包含数据准备和模型训练操作的单个管道更轻松。

​	使用单独的数据准备管道和模型时，流程与单个管道相同；但现在需要同时保存和加载两个管道。

### 保存数据准备管道和经过训练的模型

### 加载数据准备管道和经过训练的模型



# 使用经过训练的模型进行预测

## 单一预测

​	若要进行单一预测，请使用加载的预测管道创建 `PredictionEngine`。

```csharp
PredictionEngine<InModel, OutModel> predictionEngine = mlContext.Model.CreatePredictionEngine<InModel, OutModel>(predictionTransformer);
```

​	然后，使用 `Predict` 方法并将输入数据作为参数传入。 请注意，使用 `Predict` 方法不要求输入为 `IDataView`。 这是因为它可以方便地内在化输入数据类型操作，以便能够传入输入数据类型的对象。

## 批量预测

```csharp
// 准备输入对象
InModel[] inputDatas = new []{};
// 使用模型转换
IDataView predictions = predictionPipeline.Transform(inputDatas);
// 获取预测结果
float[] scoreColumn = predictions.GetColumn<float>("Score").ToArray();
```



# 重新训练模型

​	这个世界和它周围的数据在不断变化。 因此，模型也需要更改和更新。 借助 ML.NET 提供的功能，可以将已学习的模型参数作为起点并不断汲取以往经验来重新训练模型，而不必每次都从头开始。

​	以下算法可在 ML.NET 中重新训练：

- AveragedPerceptronTrainer
- FieldAwareFactorizationMachineTrainer
- LbfgsLogisticRegressionBinaryTrainer
- LbfgsMaximumEntropyMulticlassTrainer
- LbfgsPoissonRegressionTrainer
- LinearSvmTrainer
- OnlineGradientDescentTrainer
- SgdCalibratedTrainer
- SgdNonCalibratedTrainer
- SymbolicSgdLogisticRegressionBinaryTrainer

## 加载预先训练的模型

```csharp
Helper.PrintLine("重新训练模型项目");
MLContext mlContext = new MLContext();

Helper.PrintLine("加载数据处理管道和神经网络模型...");
ITransformer dataPrepPipeline = mlContext.Model.Load(DataPipelinePath, out DataViewSchema dataPrepPipelineSchema);
ITransformer trainedModel = mlContext.Model.Load(ModelPath, out DataViewSchema modelSchema);
```

## 提取预先训练的模型参数

​	加载模型后，通过访问预先训练模型的 `Model` 属性来提取已学习的模型参数。 使用线性回归模型 `OnlineGradientDescentTrainer` 训练预先训练的模型，该线性回归模型可创建输出 `LinearRegressionModelParameters` 的 `RegressionPredictionTransformer`]。 这些线性回归模型参数包含模型已学习的偏差和权重或系数。 这些值将用作新的重新训练模型的起点。

```csharp
LinearRegressionModelParameters originalMP =
	((ISingleFeaturePredictionTransformer<object>)trainedModel).Model as LinearRegressionModelParameters;
```

## 重新训练模型

​	重新训练模型的过程与训练模型的过程没有什么不同。 唯一的区别是，除了数据之外，[`Fit`](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.onlinelineartrainer-2.fit) 方法还将原始学习模型参数作为输入，并将它们用作重新训练过程的起点。

```csharp
 HousingData[] housingData = new []{};
 IDataView newData = mlContext.Data.LoadFromEnumerable(housingData);
 IDataView transformedNewData = dataPrepPipeline.Transform(newData);

RegressionPredictionTransformer<LinearRegressionModelParameters> retrainedModel =
	mlContext.Regression.Trainers.OnlineGradientDescent()
		.Fit(transformedNewData, originalMP);
```

## 比较模型参数

```csharp
LinearRegressionModelParameters retrainedMP = retrainedModel.Model as LinearRegressionModelParameters;

Helper.PrintLine($"比较模型参数变化：\n\t源模型参数\t|更新模型参数\t|变化\n\t{string.Join("\n\t", originalMP.Weights.Append(originalMP.Bias).Zip(retrainedMP.Weights.Append(retrainedMP.Bias)).Select(weights => $"{weights.First:F2}\t|{weights.Second:F2}\t|{weights.Second - weights.First:F2}"))}");
```



# 在 ASP.NET Core Web API 中部署模型

## 创建 ASP.NET Core Web API 项目

​	安装“Microsoft.ML NuGet 包；

​	安装 **Microsoft.Extensions.ML Nuget 包；**

​	将神经网络模型文件添加到项目，并将复”制到输出目录”的值更改为“如果较新则复制”；

## 注册 PredictionEnginePool 用于应用程序

​	若要进行单个预测，请使用 `PredictionEngine`。 若要在应用程序中使用 `PredictionEngine`，则必须在需要时创建它。 在这种情况下，可考虑的最佳做法是依赖项注入。

```csharp
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.ML;

public void ConfigureServices(IServiceCollection services)
{
    services.AddPredictionEnginePool<InModel, OutModel>()
        .FromFile("model.zip");
}
```

> `PredictionEngine` 不是线程安全类型。 为了提高性能和线程安全性，请使用 `PredictionEnginePool` 服务，该服务可创建 `PredictionEngine` 对象的 `ObjectPool` 供应用程序使用。

## 创建预测控制器

```csharp
using Microsoft.Extensions.ML;

public class PredictController : ControllerBase
{
    private readonly PredictionEnginePool<InModel, OutModel> _predictionEnginePool;

    public PredictController(PredictionEnginePool<InModel,OutModel> predictionEnginePool)
    {
        _predictionEnginePool = predictionEnginePool;
    }

    [HttpPost]
    public ActionResult<string> Post([FromBody] InModel input)
    {
        if(!ModelState.IsValid)
        {
            return BadRequest();
        }

        OutModel prediction = _predictionEnginePool.Predict(input);

        return Ok(prediction);
    }
}
```



# 使用 Infer.NET 和概率性编程创建游戏匹配列表

## 什么是概率性编程

​	借助概率性编程，可创建真实过程的统计模型。

## 安装 Infer.NET 包

​	需安装 `Microsoft.ML.Probabilistic.Compiler` 包才能使用 Infer.NET。



# 机器学习重要术语词汇表

## 准确性

​	在分类中，准确性是正确分类的项数目除以测试集内的项总数。 范围从 0（最不准确）到 1（最准确）。 准确性是模型性能的评估指标之一。 将其与精度、撤回 和 F 分数 结合考虑。

## 曲线下面积 (AUC)

​	二元分类中的一项评估指标，即曲线下面积值，它绘制真阳性率（y 轴）与误报率（x 轴）进行对照。 范围从 0.5（最差）到 1（最佳）。 也称为 ROC 曲线下面积，即，接受者操作特征曲线。

## 二元分类

​	一个分类事例，其中标签仅为两个类中的一个。

## 校准

​	校准是将原始分数映射到类成员身份的过程，用于二元和多类分类。 一些 ML.NET 训练程序的后缀为 `NonCalibrated`。 这些算法会生成一个原始分数，该分数之后必须映射到类概率。

## Catalog

​	在 ML.NET 中，目录是扩展函数的集合，按常见用途进行分组。

## 分类

​	当使用这些数据来预测某一类别，监管式机器学习任务被称为“分类”。

​	二元分类指的是仅预测两个类别（例如，将图像划分为“猫”或“狗”图片）。 多类分类指的是预测多个类别（例如，当将图像划分为特定品种狗的图片）。

## 决定系数

​	回归中的一项评估指标，表明数据与模型的匹配程度。 范围从 0 到 1。 值 0 表示数据是随机的，否则就无法与模型相匹配。 值 1 表示模型与数据完全匹配。 这通常称为 ^2、R^2 或 r 平方值。

## 数据

​	数据是所有机器学习应用程序的核心。 在 ML.NET 中，数据由 [IDataView](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.idataview) 对象表示。 数据视图对象：

- 由列和行组成
- 延迟计算，即它们仅在操作调用数据时加载数据
- 包含定义了每个列的类型、格式和长度的架构

## 估算器

​	ML.NET 中实现 IEstimator 接口的类。

​	估算器是一种转换（数据准备转换和机器学习模型训练转换）规范。 估算器可以链接在一起形成转换管道。 调用 Fit 时，会学习估算器或估算器管道的参数。 Fit 的结果为转换器。

## 扩展方法

​	一种 .NET 方法，它是类的一部分，但在类外部定义。 扩展方法的第一个参数是对扩展方法所属的类的静态 `this` 引用。

## 功能

​	正在对其进行度量的现象的一个可度量属性，通常是一个数（双精度）值。 多个特征被称为“特征向量”且通常存储为 `double[]`。 这些特征定义所度量现象的重要特性。

## 特征工程

​	特征工程是涉及定义一组特征和开发软件以从可用现象数据中生成特征向量（即特征提取）的过程。

## F 分数

​	分类中的一项评估指标，它平衡精度和撤回。

## 超参数

​	机器学习算法的参数。 示例包括在决策林中学习的树的数量，或者梯度下降算法中的步长。 在对模型进行定型之前，先设置超参数的值，并控制查找预测函数参数的过程，例如，决策树中的比较点或线性回归模型中的权重。

## Label

​	使用机器学习模型进行预测的元素。 例如，狗的品种或将来的股票价格。

## 对数损失

​	在分类中，描述分类器准确性的评估指标。 对数损失越小，分类器越准确。

## 损失函数

​	损失函数是指训练标签值与模型所做预测之间的差异。 通过最小化损失函数来估算模型参数。

​	可以为不同的训练程序配置不同的损失函数。

## 平均绝对误差 (MAE)

​	回归中的一项评估指标，即所有模型误差的平均值，其中模型误差是预测标签值和正确标签值之间的差距。

## 模型

​	就传统意义而言，它是预测函数的参数。 例如，线性回归模型中的权重或决策树中的拆分点。 在 ML.NET 中，一个模型包含预测域对象标签所需的所有信息（例如，图像或文本）。 这意味着 ML.NET 模型包括所需的特征化步骤以及预测函数参数。

## 多类分类

​	一个分类事例，其中标签是三个或更多类中的一个。

## N 元语法

​	文本数据的特征提取方案：N 个单词的任何序列都将转变为特征值。

## 数字特征向量

​	只包含数值的特征向量。 这与 `double[]` 非常类似。

## 管道

​	要将模型与数据集相匹配所需的所有操作。 管道由数据导入、转换、特征化和学习步骤组成。 对管道进行定型后，它会转变为模型。

## 精度

​	在分类中，类的精度是正确预测为属于该类的项目的数量，除以预测为属于该类的项目的总数。

## 撤回

​	在分类中，类的撤回是正确预测为属于该类的项目的数量，除以实际属于该类的项目的总数。

## 正则化

​	正则化会对过于复杂的线性模型进行惩罚。 正则化有两种类型：

- L1L1 正则化将无意义特征的权重归零。 进行这种正则化之后，所保存模型的大小可能会变小。
- L2L2 正则化将无意义特征的权重范围最小化，这是一个更常规的过程，对离群值的敏感度也较低。

## 回归测试

​	监管式机器学习任务，其中输出是一个实际值，例如，双精度值。 示例包括预测股票价格。

​	回归中的一项评估指标，即所有绝对误差总和除以正确标签值和所有正确标签值的平均值之间的差值总和。

## 相对平方误差

​	回归中的一项评估指标，即所有绝对平方误差总和除以正确标签值和所有正确标签值的平均值之间的平方差值总和。

## 均方误差根 (RMSE)

​	回归中的一项评估指标，即误差平方平均值的平方根。

## 监管式机器学习

​	机器学习的一个子类，其中所需的模型预测尚不可见的数据标签。 示例包括分类、回归以及结构化预测。

## 训练

​	识别给定定型数据集模型的过程。 对于线性模型，这意味着查找权重。 有关树信息，这涉及到标识拆分点。

## 转换器

​	一个实现 ITransformer 接口的 ML.NET 类。

​	转换器可将一个 IDataView 转换为另一个 IDataView。 转换器是通过训练估算器或估算器管道创建的。

## 非监管式机器学习

​	机器学习的子类，其中所需的模型查找数据中的隐藏（或潜在）结构。 示例包括聚类分析、主题建模和维数约简。



# ML.NET 中的机器学习任务

## 二元分类

​	监管式机器学习任务，用于预测数据实例所属的两个类（类别）。 分类算法输入是一组标记示例，其中每个标签为整数 0 或 1。 二元分类算法的输出是一个分类器，可用于预测未标记的新实例的类。二元分类方案示例包括：

- 情绪“正面”或“负面”。
- 诊断患者是否患有某种疾病。
- 决定是否要将电子邮件标记为“垃圾邮件”。
- 确定照片是否包含狗或水果。

### 二元分类训练程序

可以使用以下算法训练二元分类模型：

- [AveragedPerceptronTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.averagedperceptrontrainer)
- [SdcaLogisticRegressionBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcalogisticregressionbinarytrainer)
- [SdcaNonCalibratedBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcanoncalibratedbinarytrainer)
- [SymbolicSgdLogisticRegressionBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.symbolicsgdlogisticregressionbinarytrainer)
- [LbfgsLogisticRegressionBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lbfgslogisticregressionbinarytrainer)
- [LightGbmBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lightgbm.lightgbmbinarytrainer)
- [FastTreeBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fasttreebinarytrainer)
- [FastForestBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fastforestbinarytrainer)
- [GamBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.gambinarytrainer)
- [FieldAwareFactorizationMachineTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fieldawarefactorizationmachinetrainer)
- [PriorTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.priortrainer)
- [LinearSvmTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.linearsvmtrainer)

### 二元分类输入和输出

​	为了通过二元分类获得最佳结果，应平衡训练数据（即正训练数据和负训练数据的数量相等）。 应在训练前处理缺失值。

| 输出列名称       | 列名称                                                       | 说明                                                         |
| :--------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `Score`          | [Single](https://docs.microsoft.com/zh-cn/dotnet/api/system.single) | 由模型计算得出的原始分数                                     |
| `PredictedLabel` | [Boolean](https://docs.microsoft.com/zh-cn/dotnet/api/system.boolean) | 预测的标签，基于分数符号。 负分数映射到 `false`，正分数映射到 `true`。 |

## 多类分类

​	[监管式机器学习](https://docs.microsoft.com/zh-cn/dotnet/machine-learning/resources/glossary#supervised-machine-learning)任务，用于预测数据实例的类（类别）。 分类算法输入是一组标记示例。 每个标签通常以文本形式开始。 然后通过 TermTransform 运行，它可将其转换为 Key（数字）类型。 分类算法的输出是一个分类器，可用于预测未标记的新实例的类。 多类分类方案示例包括：

- 确定狗的品种等。
- 了解电影评论是“正面”、“中立”还是“负面”。
- 将酒店评语分类为“位置”、“价格”、“整洁度”等。

> 一个与所有升级任何二元分类学习器，以便对多类数据集进行操作。

### 多类分类训练程序

可以使用以下训练算法训练多类分类模型：

- [LightGbmMulticlassTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lightgbm.lightgbmmulticlasstrainer)
- [SdcaMaximumEntropyMulticlassTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcamaximumentropymulticlasstrainer)
- [SdcaNonCalibratedMulticlassTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcanoncalibratedmulticlasstrainer)
- [LbfgsMaximumEntropyMulticlassTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lbfgsmaximumentropymulticlasstrainer)
- [NaiveBayesMulticlassTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.naivebayesmulticlasstrainer)
- [OneVersusAllTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.oneversusalltrainer)
- [PairwiseCouplingTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.pairwisecouplingtrainer)

### 多类分类输入和输出

输入标签列数据必须为 [key](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.data.keydataviewtype) 类型。 特征列必须为 [Single](https://docs.microsoft.com/zh-cn/dotnet/api/system.single) 的固定大小向量。

| 输出名称         | 类型                                                         | 说明                                                         |
| :--------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `Score`          | [Single](https://docs.microsoft.com/zh-cn/dotnet/api/system.single)的向量 | 所有类的分数。 值越高意味着落入相关类的概率越高。 如果第 i 个元素具有最大值，则预测的标签索引为 i。 请注意，i 是从零开始的索引。 |
| `PredictedLabel` | [key](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.data.keydataviewtype)类型 | 预测标签的索引。 如果其值为 i，则实际标签为键值输入标签类型中的第 i 个类别。 |

## 回归测试

​	监管式机器学习任务，用于从一组相关特征中预测标签值。 标签可以是任何实际值，而不是像在分类任务中那样来自一组有限的值。 回归算法模拟其相关特征上的标签依赖关系，以确定标签将如何随着特征值的变化而变化。 回归算法输入是一组带已知值标签的示例。 回归算法输出是一个函数，可用于预测任何一组新输入特征的标签值。 回归方案示例包括：

- 基于房子特性（如卧室数量、位置或大小）来预测房价。
- 基于历史数据和当前市场趋势预测将来的股票价格。
- 基于广告预算预测产品销售。

### 回归训练程序

​	可以使用以下算法训练回归模型：

- [LbfgsPoissonRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lbfgspoissonregressiontrainer)
- [LightGbmRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lightgbm.lightgbmregressiontrainer)
- [SdcaRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcaregressiontrainer)
- [OlsTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.olstrainer)
- [OnlineGradientDescentTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.onlinegradientdescenttrainer)
- [FastTreeRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fasttreeregressiontrainer)
- [FastTreeTweedieTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fasttreetweedietrainer)
- [FastForestRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fastforestregressiontrainer)
- [GamRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.gamregressiontrainer)

### 回归输入和输出

​	输入标签列数据必须为 Single。

| 输出名称 | 类型                                                         | 说明               |
| :------- | :----------------------------------------------------------- | :----------------- |
| `Score`  | [Single](https://docs.microsoft.com/zh-cn/dotnet/api/system.single) | 模型预测的原始分数 |

## 聚类分析

​	非监管式机器学习任务，用于将数据实例分组到包含类似特性的群集。 聚类分析还可用来识别可能无法通过浏览或简单的观察以逻辑方式推导出的数据集中的关系。 聚类分析算法的输入和输出取决于选择的方法。 可以采取分发、质心、连接或基于密度的方法。 ML.NET 当前支持使用 K 平均值聚类分析的基于质心的方法。 聚类分析方案示例包括：

- 基于酒店选择的习惯和特征来了解酒店来宾群。
- 确定客户群和人口统计信息来帮助生成目标广告活动。
- 基于生产指标对清单进行分类。

### 聚类分析训练程序

​	可以使用以下算法训练聚类分析模型：

- [KMeansTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.kmeanstrainer)

### 聚类分析输入和输出

​	输入特征数据必须为 Single。 无需标签。

| 输出名称         | 类型                                                         | 说明                             |
| :--------------- | :----------------------------------------------------------- | :------------------------------- |
| `Score`          | [Single](https://docs.microsoft.com/zh-cn/dotnet/api/system.single) 的向量 | 给定数据点到所有群集的质心的距离 |
| `PredictedLabel` | [key](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.data.keydataviewtype) 类型 | 模型预测的最接近的群集的索引。   |

## 异常情况检测

​	此任务使用主体组件分析 (PCA) 创建异常情况检测模型。 基于 PCA 的异常情况检测有助于在以下场景中构建模型：可以很轻松地从一个类中获得定型数据（例如有效事务），但难以获得目标异常的足够示例。

​	PCA 是机器学习中已建立的一种技术，由于它揭示了数据的内部结构，并解释了数据中的差异，因此经常被用于探索性数据分析。 PCA 的工作方式是通过分析包含多个变量的数据。 它查找变量之间的关联性，并确定最能捕捉结果差异的值的组合。 这些组合的特性值用于创建一个更紧凑的特性空间，称为主体组件。

​	异常情况检测包含机器学习中的许多重要任务：

- 识别潜在的欺诈交易。
- 指示发生了网络入侵的学习模式。
- 发现异常的患者群集。
- 检查输入系统的值。

​	根据定义，异常情况属于罕见事件，因此很难收集具有代表性的数据样本用于建模。 此类别中包含的算法是专门设计用来解决使用不平衡数据集建立和定型模型的核心挑战。

### 异常情况检测训练程序

​	可以使用以下算法训练异常情况检测模型：

- [RandomizedPcaTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.randomizedpcatrainer)

### 异常情况检测输入和输出

​	输入特征必须为 Single 的固定大小向量。

| 输出名称 | 类型                                                         | 说明                                     |
| :------- | :----------------------------------------------------------- | :--------------------------------------- |
| `Score`  | [Single](https://docs.microsoft.com/zh-cn/dotnet/api/system.single) | 由异常情况检测模型计算得出的非负无界分数 |

## 排名

​	排名任务从一组标记的示例构建排名程序。 该示例集由实例组组成，这些实例组可以使用给定的标准进行评分。 每个实例的排名标签是 { 0, 1, 2, 3, 4 }。 排名程序定型为用每个实例的未知分数对新实例组进行排名。 ML.NET 排名学习器基于机器已学习的排名。

### 排名训练算法

​	可以使用以下算法训练排名模型：

- [LightGbmRankingTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lightgbm.lightgbmrankingtrainer)
- [FastTreeRankingTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fasttreerankingtrainer)

### 排名输入和输出

​	输入标签数据类型必须为 key 类型或 Single。 标签的值决定相关性，其中较高的值表示较高的相关性。 如果标签为 key 类型，则键索引为相关性值，其中最小索引是最不相关的。 如果标签为 Single，则较大的值表示较高的相关性。

​	特征数据必须为 Single 的固定大小向量，输入行组列必须为 key 类型。

| 输出名称 | 类型                                                         | 说明                           |
| :------- | :----------------------------------------------------------- | :----------------------------- |
| `Score`  | [Single](https://docs.microsoft.com/zh-cn/dotnet/api/system.single) | 由模型计算以确定预测的无界分数 |

## 建议

​	推荐任务支持生成推荐产品或服务的列表。 ML.NET 使用矩阵因子分解 (MF)，这是一协作筛选算法，当目录中有历史产品评级数据时，推荐使用该算法。 例如，你为用户提供历史电影评级数据，并希望向他们推荐接下来可能观看的其他电影。

### 建议训练算法

​	可以使用以下算法训练建议模型：

- [MatrixFactorizationTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.matrixfactorizationtrainer)



# 如何选择 ML.NET 算法

​	对于每个 ML.NET 任务，有多种训练算法可供选择。 选择哪个算法取决于尝试解决的问题、数据的特征以及可用的计算和存储资源。 **值得注意的是，训练机器学习模型是一个迭代过程。 可能需要尝试多种算法才能找到效果最好的算法。**

​	算法在**特征**上运行。 特征是根据输入数据进行计算的数字值。 它们是机器学习算法的最佳输入。 可以使用一个或多个数据转换将原始输入数据转换为特征。 例如，文本数据被转换为一组字词计数和字词组合计数。 使用数据转换从原始数据类型中提取特征后，它们被称为**特征化**。 例如，特征化文本或特征化图像数据。

## 训练程序 = 算法 + 任务

​	算法是执行后可生成**模型**的数学运算。 不同的算法生成具有不同特征的模型。

## 线性算法

​	线性算法生成一个模型，该模型根据输入数据和一组**权重**的线性组合计算**分数**。 权重是训练期间估算的模型参数。

​	线性算法适用于线性可分的特征。

​	**在使用线性算法进行训练之前，应对特征进行规范化**。 这样可防止某个特征对结果产生比其他特征更多的影响。

​	一般而言，**线性算法可缩放且速度快，训练和预测费用也很低**。 它们按特征数量进行缩放，并按训练数据集的大小粗略进行缩放。

**线性训练程序**

| 算法             | 属性                                                         | 训练程序                                                     |
| :--------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 平均感知器       | 最适合用于文本分类                                           | [AveragedPerceptronTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.averagedperceptrontrainer) |
| 随机双坐标上升   | 默认性能良好，不需要调整                                     | [SdcaLogisticRegressionBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcalogisticregressionbinarytrainer)、[SdcaNonCalibratedBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcanoncalibratedbinarytrainer)、[SdcaMaximumEntropyMulticlassTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcamaximumentropymulticlasstrainer)、[SdcaNonCalibratedMulticlassTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcanoncalibratedmulticlasstrainer) [SdcaRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.sdcaregressiontrainer) |
| L-BFGS           | 在有大量特征时使用。 生成逻辑回归训练统计数据，但缩放性能不如 AveragedPerceptronTrainer | [LbfgsLogisticRegressionBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lbfgslogisticregressionbinarytrainer)、[LbfgsMaximumEntropyMulticlassTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lbfgsmaximumentropymulticlasstrainer)、[LbfgsPoissonRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lbfgspoissonregressiontrainer) |
| 符号随机梯度下降 | 最快速、最准确的线性二元分类训练程序。 可随处理器数量很好地缩放 | [SymbolicSgdLogisticRegressionBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.symbolicsgdlogisticregressionbinarytrainer) |

## 决策树算法

​	决策树算法创建包含一系列决策的模型：实际上是包含数据值的流程图。

​	特征不需要线性可分即可使用此类算法。 特征无需规范化，因为特征向量中的各个值在决策过程中独立使用。

​	**决策树算法通常非常准确。**

​	除广义加性模型 (GAM) 之外，在存在大量特征的情况下，树模型可能缺少可解释性。

​	**决策树算法需要更多资源，并且缩放性能不如线性算法**。 它们在适用于内存的数据集中拥有良好性能。

​	提升决策树是一组小型决策树，其中每个决策树对输入数据进行评分并将分数传递到下一个决策树来生成更好的分数，以此类推，其中每个决策树都会在之前决策树的基础上有所改进。

**决策树训练程序**

| 算法               | 属性                                                         | 训练程序                                                     |
| :----------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 轻型梯度增强机     | 最快速、最准确的二元分类树训练程序。 高度可调                | [LightGbmBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lightgbm.lightgbmbinarytrainer)、[LightGbmMulticlassTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lightgbm.lightgbmmulticlasstrainer)、[LightGbmRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lightgbm.lightgbmregressiontrainer)、 [LightGbmRankingTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lightgbm.lightgbmrankingtrainer) |
| 快速决策树         | 用于特征化图像数据。 在非均衡数据方面具有弹性。 高度可调     | [FastTreeBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fasttreebinarytrainer)、[FastTreeRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fasttreeregressiontrainer)、[FastTreeTweedieTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fasttreetweedietrainer)、[FastTreeRankingTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fasttreerankingtrainer) |
| 快速林             | 适用于干扰性数据                                             | [FastForestBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fastforestbinarytrainer)、[FastForestRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fastforestregressiontrainer) |
| 广义加性模型 (GAM) | 最适合用于使用决策树算法时表现良好但可解释性为优先事项的问题 | [GamBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.gambinarytrainer)、[GamRegressionTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.gamregressiontrainer) |

## 矩阵分解

| 属性                                   | 训练程序                                                     |
| :------------------------------------- | :----------------------------------------------------------- |
| 最适合用于具有大型数据集的稀疏分类数据 | [FieldAwareFactorizationMachineTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fieldawarefactorizationmachinetrainer) |

## 元算法

这些训练程序根据二元训练程序创建多类训练程序。 与 [AveragedPerceptronTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.averagedperceptrontrainer)、[LbfgsLogisticRegressionBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lbfgslogisticregressionbinarytrainer)、[SymbolicSgdLogisticRegressionBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.symbolicsgdlogisticregressionbinarytrainer)、[LightGbmBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.lightgbm.lightgbmbinarytrainer)、[FastTreeBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fasttreebinarytrainer)、[FastForestBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.fastforestbinarytrainer)、[GamBinaryTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.fasttree.gambinarytrainer) 配合使用。

| 算法     | 属性                                                         | 训练程序                                                     |
| :------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 一对多   | 此多类分类器为每个类训练一个二元分类器，这可将该类与所有其他类区分开来。规模因要分类的类的数量而受到限制 | [OneVersusAllTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.oneversusalltrainer) |
| 成对耦合 | 此多类分类器在每对类上训练二元分类算法。 规模因类的数量而受到限制，因为必须训练每个两个类的组合。 | [PairwiseCouplingTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.pairwisecouplingtrainer) |

## K-Means

| 属性         | 训练程序                                                     |
| :----------- | :----------------------------------------------------------- |
| 用于聚类分析 | [KMeansTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.kmeanstrainer) |

## 主体组件分析

| 属性             | 训练程序                                                     |
| :--------------- | :----------------------------------------------------------- |
| 用于异常情况检测 | [RandomizedPcaTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.randomizedpcatrainer) |

## 朴素贝叶斯

| 属性                                                     | 训练程序                                                     |
| :------------------------------------------------------- | :----------------------------------------------------------- |
| 当特征独立且训练数据集很小时，请使用此多类分类训练程序。 | [NaiveBayesMulticlassTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.naivebayesmulticlasstrainer) |

## 前期训练程序

| 属性                                                         | 训练程序                                                     |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| 使用此二元分类训练程序来确定其他训练程序的性能基线。 其他训练程序的指标应优于前期训练程序才能成为有效指标。 | [PriorTrainer](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.trainers.priortrainer) |



# 数据转换

​	数据转换用于为定型模型准备数据。

​	一些数据转换需要通过定型数据来计算参数。需要先用数据训练管道，再用管道处理数据。

​	另一些数据转换不需要定型数据。

## 列映射和分组

| Transform                                                    | 定义                               |
| :----------------------------------------------------------- | :--------------------------------- |
| [Concatenate](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.transformextensionscatalog.concatenate) | 将一个或多个输入列连接到新输出列中 |
| [CopyColumns](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.transformextensionscatalog.copycolumns) | 复制和重命名一个或多个输入列       |
| [DropColumns](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.transformextensionscatalog.dropcolumns) | 删除一个或多个输入列               |
| [SelectColumns](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.transformextensionscatalog.selectcolumns) | 选择一个或多个不包含输入数据的列   |

## 规范化和缩放

| Transform                                                    | 定义                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [NormalizeMeanVariance](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.normalizationcatalog.normalizemeanvariance) | 减去（定型数据的）平均值，再除以（定型数据的）方差           |
| [NormalizeLogMeanVariance](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.normalizationcatalog.normalizelogmeanvariance) | 根据定型数据的对数进行规范化                                 |
| [NormalizeLpNorm](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.normalizationcatalog.normalizelpnorm) | 按 [lp 范数](https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions)缩放输入向量，其中 p 为 1、2 或无穷大。 默认为 l2（欧几里得距离）范数 |
| [NormalizeGlobalContrast](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.normalizationcatalog.normalizeglobalcontrast) | 缩放行中的每个值，具体方法是减去行数据的平均值，除以（行数据的）标准差或 l2 范数，再乘以可配置的比例因子（默认值为 2） |
| [NormalizeBinning](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.normalizationcatalog.normalizebinning) | 将输入值分配到箱索引，并除以箱数量，以生成介于 0 和 1 之间的浮点值。计算箱边界是为了在各个箱中均匀分布定型数据 |
| [NormalizeSupervisedBinning](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.normalizationcatalog.normalizesupervisedbinning) | 根据与标签列的相关性，将输入值分配到箱                       |
| [NormalizeMinMax](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.normalizationcatalog.normalizeminmax) | 按定型数据最小值和最大值的差值缩放输入                       |

## 数据类型转换

| Transform                                                    | 定义                                         |
| :----------------------------------------------------------- | :------------------------------------------- |
| [ConvertType](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.conversionsextensionscatalog.converttype) | 将输入列的类型转换为新类型                   |
| [MapValue](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.conversionsextensionscatalog.mapvalue) | 根据提供的映射字典将值映射到键（类别）       |
| [MapValueToKey](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.conversionsextensionscatalog.mapvaluetokey) | 通过从输入数据创建映射，将值映射到键（类别） |
| [MapKeyToValue](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.conversionsextensionscatalog.mapkeytovalue) | 将键转换回原始值                             |
| [MapKeyToVector](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.conversionsextensionscatalog.mapkeytovector) | 将键转换回原始值的向量                       |
| [MapKeyToBinaryVector](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.conversionscatalog.mapkeytobinaryvector) | 将键转换回原始值的二元向量                   |
| [Hash](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.conversionsextensionscatalog.hash) | 哈希处理输入列中的值                         |

## 文本转换

| Transform                                                    | 定义                                                     |
| :----------------------------------------------------------- | :------------------------------------------------------- |
| [FeaturizeText](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.featurizetext) | 将文本列转换为规范化 ngram 和 char-gram 计数的浮点数组   |
| [TokenizeIntoWords](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.tokenizeintowords) | 将一个或多个文本列拆分为各个字词                         |
| [TokenizeIntoCharactersAsKeys](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.tokenizeintocharactersaskeys) | 将一个或多个文本列拆分为关于一组主题的各个字符浮点数     |
| [NormalizeText](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.normalizetext) | 更改大小写，删除标注字符、标点符号和数字                 |
| [ProduceNgrams](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.producengrams) | 将文本列转换为一组 ngram 计数（连续单词的序列）          |
| [ProduceWordBags](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.producewordbags) | 将文本列转换为一组 ngram 向量计数                        |
| [ProduceHashedNgrams](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.producehashedngrams) | 将文本列转换为已哈希处理的 ngram 计数向量                |
| [ProduceHashedWordBags](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.producehashedwordbags) | 将文本列转换为一组已哈希处理的 ngram 计数                |
| [RemoveDefaultStopWords](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.removedefaultstopwords) | 从输入列中删除指定语言的默认停用词                       |
| [RemoveStopWords](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.removestopwords) | 从输入列中删除指定的停用词                               |
| [LatentDirichletAllocation](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.latentdirichletallocation) | 将文档（表示为浮点数向量）转换为关于一组主题的浮点数向量 |
| [ApplyWordEmbedding](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.textcatalog.applywordembedding) | 使用预定型模型将文本令牌向量转换为句向量                 |

## 图像转换

| Transform                                                    | 定义                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [ConvertToGrayscale](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.imageestimatorscatalog.converttograyscale) | 将图像转换为灰度图像                                         |
| [ConvertToImage](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.imageestimatorscatalog.converttoimage) | 将像素向量转换为 [ImageDataViewType](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.transforms.image.imagedataviewtype) |
| [ExtractPixels](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.imageestimatorscatalog.extractpixels) | 将输入图像中的像素转换为数字向量                             |
| [LoadImages](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.imageestimatorscatalog.loadimages) | 将图像从文件夹加载到内存中                                   |
| [ResizeImages](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.imageestimatorscatalog.resizeimages) | 调整图像大小                                                 |

## 分类数据转换

| Transform                                                    | 定义                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [OneHotEncoding](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.categoricalcatalog.onehotencoding) | 将一个或多个文本列转换为[单热](https://en.wikipedia.org/wiki/One-hot)编码向量 |
| [OneHotHashEncoding](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.categoricalcatalog.onehothashencoding) | 将一个或多个文本列转换为基于哈希的单热编码向量               |

## 缺失值

| Transform                                                    | 定义                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [IndicateMissingValues](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.extensionscatalog.indicatemissingvalues) | 新建布尔输出列：如果输入列中缺少值，输出列的值为 true        |
| [ReplaceMissingValues](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.extensionscatalog.replacemissingvalues) | 新建输出列：如果输入列中缺少值，输出列的值设置为默认值，否则设置为输入值 |

## 功能选择

| Transform                                                    | 定义                           |
| :----------------------------------------------------------- | :----------------------------- |
| [SelectFeaturesBasedOnCount](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.featureselectioncatalog.selectfeaturesbasedoncount) | 选择非默认值大于阈值的功能     |
| [SelectFeaturesBasedOnMutualInformation](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.featureselectioncatalog.selectfeaturesbasedonmutualinformation) | 选择标签列中的数据最依赖的功能 |

## 自定义转换

| Transform                                                    | 定义                               |
| :----------------------------------------------------------- | :--------------------------------- |
| [CustomMapping](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.custommappingcatalog.custommapping) | 使用用户定义映射将现有列转换为新列 |



# ML.NET 中的模型评估指标

## 二元分类指标

| 指标        | 说明                                                         | 期望                                                         |
| :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **准确性**  | [准确性](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification)指正确预测与测试数据集的比例。 它是正确预测次数与输入示例总数的比率。 仅当属于每个类的示例数量相似时，它才能正常工作。 | **越接近 1.00 越好**。 但刚好 1.00 表示存在问题（通常包括：标签/目标泄漏、过度拟合或使用训练数据进行测试）。 当测试数据处于非平衡状态（大部分实例属于其中一个类）、数据集非常小，或分数趋近 0.00 或 1.00 时，准确性并不能真实反应分类器的效果，此时需要检查其他指标。 |
| **AUC**     | [aucROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) 或*曲线下面积*：此指标通过扫描真正率和假正率来测量曲线下面积。 | **越接近 1.00 越好**。 应该大于 0.50 才可接受模型；AUC 为 0.50 或更小的模型毫无价值。 |
| **AUCPR**   | [aucPR](https://www.coursera.org/lecture/ml-classification/precision-recall-curve-rENu8) 或*查准率-查全率曲线的曲线下面积*：在类非常不均衡的情况下（高度偏斜的数据集），是成功预测的有用度量值。 | **越接近 1.00 越好**。 接近 1.00 的高分表明分类器返回准确的结果（高查准率），以及返回大部分正结果（高查全率）。 |
| **F1 分数** | [F1 分数](https://en.wikipedia.org/wiki/F1_score)也称为*均衡 F 分数或 F 度量值*。 这是查准率和查全率的调和平均数。 如果想要在查准率和查全率之间寻求平衡，F1 分数很有用。 | **越接近 1.00 越好**。 F1 分数的最佳值为 1.00，最差值为 0.00。它可指示分类器的查准率。 |

## 多类分类指标

| 指标             | 说明                                                         | 期望                                                         |
| :--------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **微观准确性**   | [微平均准确性](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.data.multiclassclassificationmetrics.microaccuracy#Microsoft_ML_Data_MulticlassClassificationMetrics_MicroAccuracy)聚合所有类的贡献度来计算平均指标。 它是正确预测的实例的部分。 微平均不考虑类成员资格。 基本而言，每个“示例-类”对对准确性指标的贡献度相同。 | **越接近 1.00 越好**。 在多类分类任务中，如果怀疑可能存在类不均衡的现象（即 某个类的实例可能比其他类的实例多很多），则应选择微观准确性，而不是宏观准确性。 |
| **宏观准确性**   | [宏平均准确性](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.data.multiclassclassificationmetrics.macroaccuracy#Microsoft_ML_Data_MulticlassClassificationMetrics_MacroAccuracy)指类级别的平均准确性。 每个类的准确性都会进行计算，宏观准确性就是这些准确性的平均值。 基本而言，每个类对准确性指标的贡献度相同。 占比较小的类与占比较大的类拥有同等的权重。 无论数据集包含多少个来自该类的实例，宏平均指标都会对每个类赋予相同的权重。 | **越接近 1.00 越好**。 它为每个类单独计算该指标，然后取平均值（因此实现平等对待所有类） |
| **对数损失**     | [对数损失](http://wiki.fast.ai/index.php/Log_Loss)测量分类模型的性能，其中预测输入是介于 0.00 和 1.00 之间的概率值。 随着预测概率偏离实际标签，对数损失会增加。 | **越接近 0.00 越好**。 完美模型的对数损失为 0.00。 我们的机器学习模型旨在最小化此值。 |
| **对数损失减小** | [对数损失减小](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.data.multiclassclassificationmetrics.loglossreduction#Microsoft_ML_Data_MulticlassClassificationMetrics_LogLossReduction)可以解释为分类器相较随机预测的优势。 | **取值范围为 [-inf, 1.00]，其中 1.00 表示完美的预测，0.00 表示准确性一般的预测**。 例如，如果该值等于 0.20，则可以将其解释为“正确预测的概率比随机猜测的概率高 20%” |

微观准确性通常能够更好地与 ML 预测的业务需求保持一致。 如果想要选择单个指标用于选择多类分类任务的质量，则通常应选择微观准确性。

## 回归指标

| 指标           | 说明                                                         | 期望                                                         |
| :------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **R 平方**     | [R 平方 (R2)](https://en.wikipedia.org/wiki/Coefficient_of_determination) 又称为*决定系数*，其将模型的预测能力表示为取值范围介于 -inf 和 1.00 之间的值。 1.00 意味着完美拟合，且拟合度可以无穷差，因此分数可能为负数。分数为 0.00 表示模型正在猜测标签的预期值。 R2 测量实际测试数据值与预测值的接近程度。 | **越接近 1.00 表示质量越好**。 但是，有时较低的 R 平方值（例如 0.50）对于方案而言可能完全正常或足够好，并且较高的 R 平方值并不总是好结果且有时可疑。 |
| **绝对值损失** | [绝对值损失](https://en.wikipedia.org/wiki/Mean_absolute_error)又称为*平均绝对误差 (MAE)*，其测量预测结果与实际结果的接近程度。 它是所有模型误差的平均值，其中模型误差是预测标签值和正确标签值之间的绝对差距。 为测试数据集的每个记录计算此预测误差。 最后，计算所有已记录的绝对误差的平均值。 | **越接近 0.00 表示质量越好。** 请注意，平均绝对误差使用与待测量数据相同的比例（未规范化为特定范围）。 绝对值损失、平方损失和 RMS 损失只能用于同一数据集的模型或具有类似标签值分布的数据集之间的比较。 |
| **平方损失**   | [平方损失](https://en.wikipedia.org/wiki/Mean_squared_error)又称为*均方误差 (MSE)* 或*均方偏差 (MSD)*，可指示回归线与一组测试数据值的接近程度。 它通过获取从点到回归线的距离（这些距离即为误差 E）并对这些距离求平方来做到这一点。 求平方可赋予较大的差异更大的权重。 | 它始终是非负值，并且**越接近 0.00 的值越好**。 可能会无法获得非常小的均方误差值，具体取决于数据。 |
| **RMS 损失**   | [RMS 损失](https://en.wikipedia.org/wiki/Root-mean-square_deviation)又称为*均方根误差 (RMSE)* 和 *均方根偏差 (RMSD)*，其测量模型预测的值与在建模环境中实际观测到的值之间的差异。 RMS 损失是平方损失的平方根，其具有与标签相同的单位，类似于绝对值损失，但赋予了较大的差异更大的权重。 均方根误差通常用于气候学、预测和回归分析，以验证试验结果。 | 它始终是非负值，并且**越接近 0.00 的值越好**。 RMSD 是准确性度量值，用于比较特定数据集的不同模型的预测误差，而不用于比较数据集之间的预测误差，因为它与比例相关。 |



# 提高 ML.NET 模型准确性

## 重新定义问题

​	有时，改进模型可能与用于训练模型的数据或技术无关。 相反，可能只是提出了错误的问题。 考虑从不同的角度看待问题，并利用数据提取潜在指示和隐藏关系，以便优化问题。

## 提供更多数据示例

​	与人类一样，训练算法获得的信息越多，性能更好的可能性就越大。 提高模型性能的方法之一是为算法提供更多训练数据示例。 模型从中进行学习的数据越多，模型能够正确识别的案例就越多。

## 为数据添加上下文

​	单个数据点的含义可能难以解读。 围绕数据点生成上下文有助于算法和主题专家更好地做出决策。例如，房屋拥有三间卧室这一事实本身并不能很好地说明其价格。 但是，如果在添加上下文之后，现在知道它位于大都市圈附近的郊区，屋主平均年龄为 38 岁，家庭平均收入为 80,000 美元，附近学校的排名在前 20%，那么算法可在更多信息的基础之上做出自己的决策。 所有这些上下文都可以作为特征添加到机器学习模型的输入中。

## 使用有意义的数据和特征

​	虽然更多的数据示例和特征可以帮助提高模型的准确性，但它们也可能会引入干扰，因为并非所有数据和特征都有意义。 因此，了解哪些特征对算法所做的决策具有最大影响非常重要。 使用排列特征重要性 (PFI) 等技术可以帮助识别这些突出的特征，这不仅有助于解释模型，还可以使用输出作为特征选择方法来减少进入训练过程的干扰特征的数量。

## 交叉验证

​	交叉验证是一种训练和模型评估技术，可将数据拆分为多个分区，并利用这些分区训练多个算法。此技术通过保留来自训练过程的数据来提高模型的可靠性。 除提高不可见观测的性能之外，在数据受限的环境中，它还可用作使用较小数据集训练模型的有效工具。

## 超参数调整

​	训练机器学习模型是一个迭代和探索的过程。 例如，使用 K-Means 算法训练模型时，最优群集数是多少？ 答案取决于许多因素，例如数据的结构。 找到该数字需要尝试不同的 k 值，然后通过评估性能来确定最佳值。 调整这些参数以找到最佳模型的做法称为超参数调整。

## 选择其他算法

​	回归和分类等机器学习任务包含各种算法实现。 可能出现尝试解决的问题和数据的构造方式与当前算法不能很好地匹配的情况。 在此类情况下，请考虑为任务使用其他算法，查看它是否能够从数据中更好地学习。

