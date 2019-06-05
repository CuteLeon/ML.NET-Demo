using System;
using System.IO;
using GitHubIssueClassification.Models;
using Microsoft.ML;
using ML.Utils;

namespace GitHubIssueClassification
{
    internal class Program
    {
        /// <summary>
        /// 训练数据路径
        /// </summary>
        private static readonly string TrainingDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "issues_train.tsv");

        /// <summary>
        /// 测试数据路径
        /// </summary>
        private static readonly string TestDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "issues_test.tsv");

        /// <summary>
        /// 训练模型数据路径
        /// </summary>
        private static readonly string ModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Datas", "Model.zip");

        private static void Main(string[] args)
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
                IDataView trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(TrainingDataPath, hasHeader: true);

                // 创建神经网络管道
                Helper.PrintLine("创建神经网络管道...");
                IEstimator<ITransformer> pipeline = CreatePipeline(mlContext);
                var trainingPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                // 开始训练神经网络
                Helper.PrintSplit();
                Helper.PrintLine("开始训练神经网络...");
                model = trainingPipeline.Fit(trainingDataView);
                Helper.PrintLine("训练神经网络完成");
                Helper.PrintSplit();

                Helper.PrintLine($"导出神经网络模型...");
                mlContext.Model.Save(model, trainingDataView.Schema, ModelPath);
            }

            // 测试
            Helper.PrintLine("评估神经网络：");
            var testDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(TestDataPath, hasHeader: true);
            var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testDataView));
            Helper.PrintLine($"\t=>微观准确性：{testMetrics.MicroAccuracy:0.###}");
            Helper.PrintLine($"\t=>宏观准确性：{testMetrics.MacroAccuracy:0.###}");
            Helper.PrintLine($"\t=>对数损失：{testMetrics.LogLoss:#.###}");
            Helper.PrintLine($"\t=>对数损失减小：{testMetrics.LogLossReduction:#.###}");

            // 预测
            Helper.PrintLine("预测：");
            Predict(mlContext, model);

            Helper.Exit(0);
        }

        /// <summary>
        /// 创建管道
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        private static IEstimator<ITransformer> CreatePipeline(MLContext mlContext)
        {
            // 绑定预测标签
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                // 绑定标题特征
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                // 绑定描述特征
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                // 绑定特征
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                // 缓存数据视图，多次使用数据时可以提高性能（适用于小中型数据集）
                .AppendCacheCheckpoint(mlContext);

            return pipeline;
        }

        /// <summary>
        /// 使用神经网络模型预测问题分类
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void Predict(MLContext mlContext, ITransformer model)
        {
            // 创建预测引擎
            Helper.PrintLine("创建预测引擎...");
            var engine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(model);

            (string Title, string Description, bool Exit) inputs;
            while (!(inputs = GetInputs()).Exit)
            {
                var issue = new GitHubIssue() { Title = inputs.Title, Description = inputs.Description };
                var prediction = engine.Predict(issue);
                Helper.PrintLine($"=> {prediction.Area}");
            }

            Console.ResetColor();
            Helper.PrintLine("结束预测");

            (string Title, string Description, bool Exit) GetInputs()
            {
                Console.ResetColor();
                Helper.PrintLine("输入问题标题以预测问题分类 (输入 exit 跳出预测)：");
                Console.Write(">>>\t请输入：");
                Console.ForegroundColor = ConsoleColor.Yellow;
                var title = Console.ReadLine();
                if (title.ToLower() == "exit")
                {
                    return (title, string.Empty, true);
                }

                Console.ResetColor();
                Helper.PrintLine("输入问题描述以预测问题分类：");
                Console.Write(">>>\t请输入：");
                Console.ForegroundColor = ConsoleColor.Yellow;
                var description = Console.ReadLine();
                Console.ForegroundColor = ConsoleColor.Magenta;

                return (title, description, false);
            }
        }
    }
}
