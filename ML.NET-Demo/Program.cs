using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using ML.NET_Demo.DataReader;
using ML.NET_Demo.Models;
using ML.NET_Demo.Utils;

namespace ML.NET_Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            var dataReader = new HouseDataReader();
            ITransformer model = null;

            if (File.Exists("model.zip"))
            {
                Helper.PrintLine("已存在模型文件 model.zip，开始加载...");
                // 加载训练模型
                model = mlContext.Model.Load("model.zip", out DataViewSchema schema);
            }
            else
            {
                // 导入训练数据
                var houses = dataReader.GetTrainingDatas().ToArray();
                Helper.PrintLine($"训练数据：\n\t{string.Join("\n\t", houses.Select(house => $"面积: {house.Size.ToString("N2")}\t价格: {house.Price}"))}");
                Helper.PrintSplit();

                IDataView trainingData = mlContext.Data.LoadFromEnumerable(houses);

                // 指定数据预备和训练管道
                /* Features: 特征 表示神经网络的输入 */
                var estimator = mlContext.Transforms
                    .Concatenate("Features", new[] { "Size" })
                    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

                // 训练模型
                Helper.PrintLine("开始训练...");
                model = estimator.Fit(trainingData);
                Helper.PrintLine("训练结束");
                Helper.PrintSplit();

                // 导出训练模型
                mlContext.Model.Save(model, trainingData.Schema, "model.zip");
            }

            // 预测
            /* Score: 分数 表示神经网络的输出 */
            Helper.PrintLine($"预测：");
            // 创建预测引擎
            var engine = mlContext.Model.CreatePredictionEngine<House, Prediction>(model);
            Enumerable.Range(10, 20).ToList().ForEach(index =>
            {
                var price = engine.Predict(new House(index, 0f));
                Helper.PrintLine($"\t面积: {index}\t价格: {price.Price}");
            });
            Helper.PrintSplit();

            Helper.PrintLine($"评估：");
            var testHouseDataView = mlContext.Data.LoadFromEnumerable(dataReader.GetTestDatas());
            var testPriceDataView = model.Transform(testHouseDataView);
            var metrics = mlContext.Regression.Evaluate(testPriceDataView, labelColumnName: "Price");
            Helper.PrintLine($"R^2: {metrics.RSquared:0.##}");
            Helper.PrintLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");

            Console.Read();
        }
    }
}
