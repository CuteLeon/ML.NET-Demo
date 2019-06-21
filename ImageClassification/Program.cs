using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ImageClassification.Models;
using Microsoft.ML;
using ML.Utils;

namespace ImageClassification
{
    class Program
    {
        static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "Datas");
        static readonly string _trainTagsTsv = Path.Combine(_assetsPath, "inputs-train", "data", "tags.tsv");
        static readonly string _predictImageListTsv = Path.Combine(_assetsPath, "inputs-predict", "data", "image_list.tsv");
        static readonly string _trainImagesFolder = Path.Combine(_assetsPath, "inputs-train", "data");
        static readonly string _predictImagesFolder = Path.Combine(_assetsPath, "inputs-predict", "data");
        static readonly string _predictSingleImage = Path.Combine(_assetsPath, "inputs-predict-single", "data", "toaster3.jpg");
        static readonly string _inceptionPb = Path.Combine(_assetsPath, "inputs-train", "inception", "tensorflow_inception_graph.pb");
        static readonly string _outputImageClassifierZip = Path.Combine(_assetsPath, "outputs", "imageClassifier.zip");
        private static string LabelTokey = nameof(LabelTokey);
        private static string PredictedLabelValue = nameof(PredictedLabelValue);

        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        static void Main(string[] args)
        {
            Helper.PrintLine("创建 MLContext...");
            MLContext mlContext = new MLContext(seed: 1);

            Helper.PrintLine("转换数据并训练模型...");
            var model = ReuseAndTuneInceptionModel(mlContext, _trainTagsTsv, _trainImagesFolder, _inceptionPb, _outputImageClassifierZip);

            Helper.PrintLine("使用已加载的模型来分类图像...");
            ClassifyImages(mlContext, _predictImageListTsv, _predictImagesFolder, _outputImageClassifierZip, model);

            Helper.PrintLine("使用已加载的模型来分类单张图像...");
            ClassifySingleImage(mlContext, _predictSingleImage, _outputImageClassifierZip, model);

            Console.Read();
        }

        /// <summary>
        /// 显示预测结果
        /// </summary>
        /// <param name="imagePredictionData"></param>
        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"图像: {Path.GetFileName(prediction.ImagePath)} 预测为: {prediction.PredictedLabelValue} 得分: {prediction.Score.Max()} ");
            }
        }

        /// <summary>
        /// 从 TSV 文件读取图像
        /// </summary>
        /// <param name="file"></param>
        /// <param name="folder"></param>
        /// <returns></returns>
        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
            => File.ReadAllLines(file)
                .Select(line => line.Split('\t'))
                .Select(line => new ImageData() { ImagePath = Path.Combine(folder, line[0]) });

        /// <summary>
        /// 转换数据并训练模型
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="dataLocation"></param>
        /// <param name="imagesFolder"></param>
        /// <param name="inputModelLocation"></param>
        /// <param name="outputModelLocation"></param>
        /// <returns></returns>
        public static ITransformer ReuseAndTuneInceptionModel(MLContext mlContext, string dataLocation, string imagesFolder, string inputModelLocation, string outputModelLocation)
        {
            // 加载数据
            var data = mlContext.Data.LoadFromTextFile<ImageData>(path: dataLocation, hasHeader: false);

            // 提取功能和转换数据
            var estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelTokey, inputColumnName: "Label")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _trainImagesFolder, inputColumnName: nameof(ImageData.ImagePath)))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(mlContext.Model.LoadTensorFlowModel(inputModelLocation)
                .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                // 选择定型算法
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelTokey, featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            // 训练
            ITransformer model = estimator.Fit(data);

            // 转换
            var predictions = model.Transform(data);

            var imageData = mlContext.Data.CreateEnumerable<ImageData>(data, false, true);
            var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);

            DisplayResults(imagePredictionData);

            // 评估模型
            var multiclassContext = mlContext.MulticlassClassification;
            var metrics = multiclassContext.Evaluate(predictions, labelColumnName: LabelTokey, predictedLabelColumnName: "PredictedLabel");
            Console.WriteLine($"对数损失: {metrics.LogLoss}");
            Console.WriteLine($"每类对数损失: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

            return model;
        }

        /// <summary>
        /// 使用已加载的模型来分类图像
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="dataLocation"></param>
        /// <param name="imagesFolder"></param>
        /// <param name="outputModelLocation"></param>
        /// <param name="model"></param>
        public static void ClassifyImages(MLContext mlContext, string dataLocation, string imagesFolder, string outputModelLocation, ITransformer model)
        {
            // 将.TSV 文件读取到 IEnumerable 中
            var imageData = ReadFromTsv(dataLocation, imagesFolder);
            var imageDataView = mlContext.Data.LoadFromEnumerable(imageData);

            // 根据测试数据预测图像分类
            var predictions = model.Transform(imageDataView);
            var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);

            DisplayResults(imagePredictionData);
        }

        /// <summary>
        /// 使用已加载的模型来分类单张图像
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="imagePath"></param>
        /// <param name="outputModelLocation"></param>
        /// <param name="model"></param>
        public static void ClassifySingleImage(MLContext mlContext, string imagePath, string outputModelLocation, ITransformer model)
        {
            var imageData = new ImageData()
            {
                ImagePath = imagePath
            };
            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);

            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
        }
    }
}
