using System;
using System.IO;
using Microsoft.ML;
using ML.Utils;
using TaxiFarePrediction.Models;

namespace TaxiFarePrediction
{
    class Program
    {
        /// <summary>
        /// 训练数据路径
        /// </summary>
        private static readonly string TrainingDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "taxi-fare-train.csv");

        /// <summary>
        /// 测试数据路径
        /// </summary>
        private static readonly string TestDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "taxi-fare-test.csv");

        /// <summary>
        /// 训练模型数据路径
        /// </summary>
        private static readonly string ModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "Model.zip");

        static void Main(string[] args)
        {
            Helper.PrintLine("创建 MLContext...");
            MLContext mlContext = new MLContext(seed: 0);
            ITransformer model = null;

            if (File.Exists(ModelPath))
            {
                Helper.PrintLine("加载神经网络模型...");
                model = mlContext.Model.Load(ModelPath, out DataViewSchema inputScema);
            }
            else
            {
                // 训练数据集合
                IDataView trainingDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TrainingDataPath, hasHeader: true, separatorChar: ',');

                // 创建神经网络管道
                Helper.PrintLine("创建神经网络管道...");
                var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                    .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                    .Append(mlContext.Regression.Trainers.FastTree());

                // 开始训练神经网络
                Helper.PrintSplit();
                Helper.PrintLine("开始训练神经网络...");
                model = pipeline.Fit(trainingDataView);
                Helper.PrintLine("训练神经网络完成");
                Helper.PrintSplit();

                Helper.PrintLine($"导出神经网络模型...");
                mlContext.Model.Save(model, trainingDataView.Schema, ModelPath);
            }

            // 测试
            Helper.PrintLine("评估神经网络：");
            var testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TestDataPath, hasHeader: true, separatorChar: ',');
            var testMetrics = mlContext.Regression.Evaluate(model.Transform(testDataView), "Label", "Score");
            Helper.PrintLine($"\t=>R^2: {testMetrics.RSquared:0.###}");
            Helper.PrintLine($"\t=>RMS error: {testMetrics.RootMeanSquaredError:0.###}");

            // 预测
            Helper.PrintLine("预测：");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };
            var prediction = predictionEngine.Predict(taxiTripSample);
            Helper.PrintLine($"预测价格: {prediction.FareAmount:0.####}, actual fare: 15.5");

            Helper.Exit(0);
        }
    }
}
