using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using ML.Utils;
using MovieRecommendation.Models;

namespace MovieRecommendation
{
    class Program
    {
        /// <summary>
        /// 训练数据路径
        /// </summary>
        private static readonly string TrainingDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "recommendation-ratings-train.csv");

        /// <summary>
        /// 测试数据路径
        /// </summary>
        private static readonly string TestingDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "recommendation-ratings-test.csv");

        /// <summary>
        /// 训练模型数据路径
        /// </summary>
        private static readonly string ModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "Model.zip");

        static void Main(string[] args)
        {
            Helper.PrintLine("创建 MLContext...");
            MLContext mlContext = new MLContext(seed: 0);
            ITransformer model;
            IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(TestingDataPath, hasHeader: true, separatorChar: ',');

            if (File.Exists(ModelPath))
            {
                Helper.PrintLine("加载神经网络模型...");
                model = mlContext.Model.Load(ModelPath, out DataViewSchema inputScema);
            }
            else
            {
                // 数据集合
                IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(TrainingDataPath, hasHeader: true, separatorChar: ',');

                // 创建神经网络管道
                Helper.PrintLine("创建神经网络管道...");
                IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion
                    .MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"))
                    .Append(mlContext.Recommendation().Trainers.MatrixFactorization(
                         new MatrixFactorizationTrainer.Options
                         {
                             MatrixColumnIndexColumnName = "userIdEncoded",
                             MatrixRowIndexColumnName = "movieIdEncoded",
                             LabelColumnName = "Label",
                             NumberOfIterations = 20,
                             ApproximationRank = 100
                         }));

                // 开始训练神经网络
                Helper.PrintSplit();
                Helper.PrintLine("开始训练神经网络...");
                model = estimator.Fit(trainingDataView);
                Helper.PrintLine("训练神经网络完成");
                Helper.PrintSplit();

                Helper.PrintLine($"导出神经网络模型...");
                mlContext.Model.Save(model, trainingDataView.Schema, ModelPath);
            }

            // 预测
            Helper.PrintLine("预测：");
            var prediction = model.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            Helper.PrintLine($"R^2: {metrics.RSquared:0.##}");
            Helper.PrintLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");

            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            var testInput = new MovieRating { userId = 6, movieId = 10 };
            var movieRatingPrediction = predictionEngine.Predict(testInput);
            if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
            {
                Helper.PrintLine($"Movie {testInput.movieId} is recommended for user {testInput.userId}");
            }
            else
            {
                Helper.PrintLine($"Movie {testInput.movieId} is not recommended for user {testInput.userId}");
            }

            Helper.Exit(0);
        }
    }
}
