using Microsoft.ML.Data;

namespace SentimentAnalysis.Models
{
    /// <summary>
    /// 情感数据
    /// </summary>
    public class SentimentData
    {
        /// <summary>
        /// 情感文本
        /// </summary>
        [LoadColumn(0)]
        public string SentimentText;

        /// <summary>
        /// 情感
        /// </summary>
        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }
}
