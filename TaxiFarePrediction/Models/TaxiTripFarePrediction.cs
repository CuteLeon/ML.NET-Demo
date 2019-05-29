using Microsoft.ML.Data;

namespace TaxiFarePrediction.Models
{
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
