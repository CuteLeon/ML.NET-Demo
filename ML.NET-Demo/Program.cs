using System;
using System.Linq;
using Microsoft.ML;
using ML.NET_Demo.DataReader;
using ML.NET_Demo.Models;

namespace ML.NET_Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // 导入训练数据
            var dataReader = new HouseDataReader();
            var houses = dataReader.GetTrainingDatas().ToArray();
            Console.WriteLine($"训练数据：\n\t{string.Join("\n\t", houses.Select(house => $"面积: {house.Size}\t价格: {house.Price}"))}");
            Console.WriteLine();

            var trainingData = mlContext.Data.LoadFromEnumerable(houses);

            // 指定数据预备和训练管道
            var pipeline = mlContext.Transforms
                .Concatenate("Features", new[] { "Size" })
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            // 训练模型
            var model = pipeline.Fit(trainingData);

            // 创建预测引擎
            var engine = mlContext.Model.CreatePredictionEngine<House, Prediction>(model);

            // 预测
            Console.WriteLine($"预测：");
            Enumerable.Range(10, 20).ToList().ForEach(index =>
            {
                var price = engine.Predict(new House(index, 0f));
                Console.WriteLine($"\t面积: {index}\t价格: {price.Price}");
            });

            Console.WriteLine($"评估：");
            var testHouseDataView = mlContext.Data.LoadFromEnumerable(dataReader.GetTestDatas());
            var testPriceDataView = model.Transform(testHouseDataView);
            var metrics = mlContext.Regression.Evaluate(testPriceDataView, labelColumnName: "Price");
            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");

            Console.Read();
        }
    }
}
