using System;
using System.IO;
using IrisFlowerClustering.Models;
using Microsoft.ML;
using ML.Utils;

namespace IrisFlowerClustering
{
    class Program
    {
        /// <summary>
        /// 训练数据路径
        /// </summary>
        private static readonly string TrainingDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "iris.data");

        /// <summary>
        /// 训练模型数据路径
        /// </summary>
        private static readonly string ModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "Model.zip");

        static void Main(string[] args)
        {
            Helper.PrintLine("创建 MLContext...");
            MLContext mlContext = new MLContext(seed: 0);
            ITransformer model;

            if (File.Exists(ModelPath))
            {
                Helper.PrintLine("加载神经网络模型...");
                model = mlContext.Model.Load(ModelPath, out DataViewSchema inputScema);
            }
            else
            {
                // 训练数据集合
                IDataView trainingDataView = mlContext.Data.LoadFromTextFile<IrisData>(TrainingDataPath, hasHeader: false, separatorChar: ',');

                // 创建神经网络管道
                Helper.PrintLine("创建神经网络管道...");
                IEstimator<ITransformer> pipeline = mlContext.Transforms
                    .Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    // 拆分为三个集群
                    .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));

                // 开始训练神经网络
                Helper.PrintSplit();
                Helper.PrintLine("开始训练神经网络...");
                model = pipeline.Fit(trainingDataView);
                Helper.PrintLine("训练神经网络完成");
                Helper.PrintSplit();

                Helper.PrintLine($"导出神经网络模型...");
                mlContext.Model.Save(model, trainingDataView.Schema, ModelPath);
            }

            IrisData setosa = new IrisData
            {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f
            };

            // 预测
            Helper.PrintLine("预测：");
            var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);
            var prediction = predictor.Predict(setosa);
            Helper.PrintLine($"所属集群: {prediction.PredictedClusterId}");
            Helper.PrintLine($"特征差距: {string.Join(" ", prediction.Distances)}");

            Helper.Exit(0);
        }
    }
}
