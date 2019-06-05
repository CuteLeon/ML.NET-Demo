using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CrossValidation.Models;
using Microsoft.ML;
using ML.Utils;

namespace CrossValidation
{
    class Program
    {
        private static readonly string TrainDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "data.csv");
        private static readonly string BestModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "BestModel.csv");

        static void Main(string[] args)
        {
            Helper.PrintLine("使用交叉验证训练模型...");
            MLContext mlContext = new MLContext();

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

            Helper.PrintLine("创建数据估算器对象...");
            IEstimator<ITransformer> sdcaEstimator = mlContext.Regression.Trainers.Sdca();

            Helper.PrintSplit();
            Helper.PrintLine("交叉验证...");
            var cvResults = mlContext.Regression.CrossValidate(transformedData, sdcaEstimator, numberOfFolds: 5);
            Helper.PrintSplit();

            Helper.PrintLine($"输出模型性能：\n\t{string.Join("\n\t", cvResults.Select(result => $">>> Fold: {result.Fold} —————— >>>\n\tRSquared: {result.Metrics.RSquared}\n\tLossFunction: {result.Metrics.LossFunction}\n\tMeanAbsoluteError: {result.Metrics.MeanAbsoluteError}\n\tMeanSquaredError: {result.Metrics.MeanSquaredError}\n\tRootMeanSquaredError: {result.Metrics.RootMeanSquaredError}"))}");
            Helper.PrintSplit();

            var bestResult = cvResults.OrderByDescending(result => result.Metrics.RSquared).First();
            Helper.PrintLine($"性能最佳模型：\n\tFold: {bestResult.Metrics.RSquared}\n\tRSquared: {bestResult.Metrics.RSquared}");
            mlContext.Model.Save(bestResult.Model, transformedData.Schema, BestModelPath);
            Helper.PrintLine($"保存性能最佳模型=> {Path.GetRelativePath(AppDomain.CurrentDomain.BaseDirectory, BestModelPath)}");

            Helper.Exit(0);
        }
    }
}
