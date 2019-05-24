using Microsoft.ML.Data;

namespace ML.NET_Demo.Models
{
    /// <summary>
    /// 预测
    /// </summary>
    public class Prediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
