using System;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using ML.Utils;
using PermutationFeatureImportance.Models;

namespace PermutationFeatureImportance
{
    class Program
    {
        private static readonly string TrainDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "data.csv");

        static void Main(string[] args)
        {
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

            ImmutableArray<RegressionMetricsStatistics> pfi = mlContext.Regression.PermutationFeatureImportance(
                sdcaModel,
                preprocessedTrainData,
                permutationCount: 3);

            Helper.PrintLine("按相关性排序特征...");
            var featureImportanceMetrics = pfi
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(myFeatures => Math.Abs(myFeatures.RSquared.Mean))
                .ToArray();

            Helper.PrintSplit();
            Helper.PrintLine($"特征 PFI:\n\t{string.Join("\n\t", featureImportanceMetrics.Select(feature => $">>> {featureColumnNames[feature.index]}\n\tMean: {feature.RSquared.Mean:F6}\n\tStandardDeviation: {feature.RSquared.StandardDeviation:F6}\n\tStandardError: {feature.RSquared.StandardError:F6}"))}");

            Helper.Exit(0);
        }
    }
}
