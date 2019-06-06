using System;
using System.IO;
using System.Linq;
using CrossValidation.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using ML.Utils;

namespace RetrainModel
{
    class Program
    {
        private static readonly string ModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "model.zip");
        private static readonly string DataPipelinePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "dataPipeline.zip");

        static void Main(string[] args)
        {
            Helper.PrintLine("重新训练模型项目");
            MLContext mlContext = new MLContext();

            Helper.PrintLine("加载数据处理管道和神经网络模型...");
            ITransformer dataPrepPipeline = mlContext.Model.Load(DataPipelinePath, out DataViewSchema dataPrepPipelineSchema);
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out DataViewSchema modelSchema);

            LinearRegressionModelParameters originalMP =
                ((ISingleFeaturePredictionTransformer<object>)trainedModel).Model as LinearRegressionModelParameters;

            Helper.PrintLine("重新训练神经网络...");
            HousingData[] housingData = new HousingData[]
            {
                new HousingData
                {
                    Size = 850f,
                    HistoricalPrices = new float[] { 150000f,175000f,210000f },
                    CurrentPrice = 205000f
                },
                new HousingData
                {
                    Size = 900f,
                    HistoricalPrices = new float[] { 155000f, 190000f, 220000f },
                    CurrentPrice = 210000f
                },
                new HousingData
                {
                    Size = 550f,
                    HistoricalPrices = new float[] { 99000f, 98000f, 130000f },
                    CurrentPrice = 180000f
                }
            };

            IDataView newData = mlContext.Data.LoadFromEnumerable(housingData);
            IDataView transformedNewData = dataPrepPipeline.Transform(newData);

            RegressionPredictionTransformer<LinearRegressionModelParameters> retrainedModel =
                mlContext.Regression.Trainers.OnlineGradientDescent()
                    .Fit(transformedNewData, originalMP);

            LinearRegressionModelParameters retrainedMP = retrainedModel.Model as LinearRegressionModelParameters;

            Helper.PrintLine($"比较模型参数变化：\n\t源模型参数\t|更新模型参数\t|变化\n\t{string.Join("\n\t", originalMP.Weights.Append(originalMP.Bias).Zip(retrainedMP.Weights.Append(retrainedMP.Bias)).Select(weights => $"{weights.First:F2}\t|{weights.Second:F2}\t|{weights.Second - weights.First:F2}"))}");

            Helper.Exit(0);
        }
    }
}
