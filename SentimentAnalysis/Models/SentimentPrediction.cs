using Microsoft.ML.Data;

namespace SentimentAnalysis.Models
{
    /// <summary>
    /// 情感预测
    /// </summary>
    public class SentimentPrediction : SentimentData
    {
        /// <summary>
        /// 预测结果
        /// </summary>
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        /// <summary>
        /// 发生概率
        /// </summary>
        public float Probability { get; set; }

        /// <summary>
        /// 得分
        /// </summary>
        public float Score { get; set; }
    }
}
