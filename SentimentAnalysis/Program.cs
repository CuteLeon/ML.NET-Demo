using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using ML.Utils;
using SentimentAnalysis.Models;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysis
{
    class Program
    {
        /// <summary>
        /// 训练数据路径
        /// </summary>
        private static readonly string TrainingDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "yelp_labelled.txt");

        /// <summary>
        /// 训练模型数据路径
        /// </summary>
        private static readonly string ModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "Model.zip");

        static void Main(string[] args)
        {
            Helper.PrintLine("检查训练数据文件...");
            if (!File.Exists(TrainingDataPath))
            {
                Helper.PrintLine("缺失训练数据文件，bye~");
                Helper.Exit(-1);
            }

            Helper.PrintLine("创建 MLContext...");
            MLContext mlContext = new MLContext();

            // 加载训练和测试数据
            Helper.PrintSplit();
            Helper.PrintLine("加载训练和测试数据...");
            TrainTestData splitDataView = LoadData(mlContext);

            // 创建和训练模型
            Helper.PrintLine("创建和训练神经网络模型...");
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            // 使用测试数据评估神经网络
            Evaluate(mlContext, model, splitDataView.TestSet);

            // 预测
            Predict(mlContext, model);

            Helper.Exit(0);
        }

        /// <summary>
        /// 使用神经网络模型预测情绪
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void Predict(MLContext mlContext, ITransformer model)
        {
            // 创建预测引擎
            var engine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            Helper.PrintLine("输入文本以预测情绪 (输入 exit 跳出预测)：");
            string input;
            while ((input = GetInput()).ToLower() != "exit")
            {
                var resultprediction = engine.Predict(new SentimentData() { SentimentText = input });
                Helper.PrintLine($"=> {(resultprediction.Prediction ? "正面" : "负面")}情绪 / 概率 = {resultprediction.Probability} 分数 = {resultprediction.Score}");
            }

            Console.ResetColor();
            Helper.PrintLine("结束预测");

            string GetInput()
            {
                Console.ResetColor();
                Console.Write(">>>\t请输入：");

                Console.ForegroundColor = ConsoleColor.Yellow;
                var read = Console.ReadLine();
                if (string.IsNullOrEmpty(read))
                {
                    read = "This was a very bad steak";
                }
                Console.ForegroundColor = ConsoleColor.Magenta;

                return read;
            }
        }

        /// <summary>
        /// 评估模型
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        /// <param name="splitTestSet"></param>
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Helper.PrintSplit();
            Helper.PrintLine($"评估神经网络质量：");
            // 准确度
            Helper.PrintLine($"\t准确度: {metrics.Accuracy:P2}");
            // 对正面类和负面类进行正确分类的置信度，应尽量接近 1
            Helper.PrintLine($"\t曲线下面积: {metrics.AreaUnderRocCurve:P2}");
            // 查准率和查全率之间的平衡关系的度量值，应尽量接近 1
            Helper.PrintLine($"\tF1评分: {metrics.F1Score:P2}");
        }

        /// <summary>
        /// 创建并训练神经网络模型
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainSet"></param>
        /// <returns></returns>
        private static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainSet)
        {
            // 将 SentimentText 成员设置为训练的 "特征"
            var estimator = mlContext.Transforms.Text
                .FeaturizeText(
                    outputColumnName: "Features",
                    inputColumnName: nameof(SentimentData.SentimentText))
            // 添加机器学习算法
            .Append(
                mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label",
                    featureColumnName: "Features"));

            // 使用训练数据训练神经网络
            Helper.PrintLine($"开始训练神经网络...");
            ITransformer model = estimator.Fit(trainSet);
            Helper.PrintLine($"神经网络训练完成");

            return model;
        }

        /// <summary>
        /// 加载训练和测试数据
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        private static TrainTestData LoadData(MLContext mlContext)
        {
            // 读取数据并拆分
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(TrainingDataPath, hasHeader: false);
            // 训练数据中取 20% 作为测试数据
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }
    }
}
