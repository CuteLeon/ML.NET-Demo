using System;
using System.IO;
using Microsoft.ML;
using ML.Utils;
using SalesAnomalyDetection.Models;

namespace SalesAnomalyDetection
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Datas", "product-sales.csv");
        const int _docsize = 36;

        static void Main(string[] args)
        {
            Helper.PrintLine("创建 MLContext ...");
            MLContext mlContext = new MLContext();

            Helper.PrintLine("加载数据 ...");
            IDataView dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: _dataPath, hasHeader: true, separatorChar: ',');

            Helper.PrintSplit();
            Helper.PrintLine("检测峰值 ...");
            DetectSpike(mlContext, _docsize, dataView);

            Helper.PrintSplit();
            Helper.PrintLine("检测更改点 ...");
            DetectChangepoint(mlContext, _docsize, dataView);

            Console.Read();
        }

        /// <summary>
        /// 检测峰值
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="docSize"></param>
        /// <param name="productSales"></param>
        private static void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95, pvalueHistoryLength: docSize / 4);
            ITransformer trainedModel = iidSpikeEstimator.Fit(productSales);
            IDataView transformedData = trainedModel.Transform(productSales);
            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            Helper.PrintLine("警报\t分数\t概率");
            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{(p.Prediction[0] == 1 ? "<-- 检测到峰值" : string.Empty)}";
                Helper.PrintLine(results);
            }
        }

        /// <summary>
        /// 检测更改点
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="docSize"></param>
        /// <param name="productSales"></param>
        private static void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95, changeHistoryLength: docSize / 4);
            var trainedModel = iidChangePointEstimator.Fit(productSales);
            IDataView transformedData = trainedModel.Transform(productSales);
            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            Helper.PrintLine("警报\t得分\t概率\t异常程度");
            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}\t{(p.Prediction[0] == 1 ? "<-- 检测到变化" : string.Empty)}";

                Helper.PrintLine(results);
            }
        }
    }
}
