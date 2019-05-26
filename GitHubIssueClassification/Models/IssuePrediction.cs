using Microsoft.ML.Data;

namespace GitHubIssueClassification.Models
{
    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }
}
