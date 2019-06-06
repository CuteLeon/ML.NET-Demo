using System;
using System.IO;
using CrossValidation.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using ML.Utils;

namespace SeparateDataPreparationAndModelPipelines
{
    class Program
    {
        private static readonly string ModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model.zip");
        private static readonly string DataPipelinePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "dataPipeline.zip");

        static void Main(string[] args)
        {
            Helper.PrintLine("使用的单独的数据准备和模型管道");
            MLContext mlContext = new MLContext();

            Helper.PrintLine("准备训练数据集...");
            HousingData[] housingData = new HousingData[]
            {
                new HousingData
                {
                    Size = 600f,
                    HistoricalPrices = new float[] { 100000f ,125000f ,122000f },
                    CurrentPrice = 170000f
                },
                new HousingData
                {
                    Size = 1000f,
                    HistoricalPrices = new float[] { 200000f, 250000f, 230000f },
                    CurrentPrice = 225000f
                },
                new HousingData
                {
                    Size = 1000f,
                    HistoricalPrices = new float[] { 126000f, 130000f, 200000f },
                    CurrentPrice = 195000f
                }
            };
            IDataView data = mlContext.Data.LoadFromEnumerable(housingData);

            Helper.PrintLine("准备数据处理管道...");
            IEstimator<ITransformer> dataPrepEstimator =
                mlContext.Transforms.Concatenate("Features", new string[] { "Size", "HistoricalPrices" })
                    .Append(mlContext.Transforms.NormalizeMinMax("Features"));
            Helper.PrintLine("训练数据处理管道...");
            ITransformer dataPrepTransformer = dataPrepEstimator.Fit(data);

            Helper.PrintLine("准备神经网络模型...");
            var sdcaEstimator = mlContext.Regression.Trainers.Sdca();
            Helper.PrintLine("训练神经网络模型...");
            IDataView transformedData = dataPrepTransformer.Transform(data);
            RegressionPredictionTransformer<LinearRegressionModelParameters> trainedModel = sdcaEstimator.Fit(transformedData);
            Helper.PrintLine("训练神经网络完成");

            Helper.PrintLine("保存数据处理管道和神经网络模型...");
            mlContext.Model.Save(dataPrepTransformer, data.Schema, DataPipelinePath);
            mlContext.Model.Save(trainedModel, transformedData.Schema, ModelPath);

            Helper.Exit(0);
        }
    }
}
