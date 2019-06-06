using Microsoft.ML.Data;

namespace PermutationFeatureImportance.Models
{
    public class HousingPriceData
    {
        [LoadColumn(0)]
        public float CrimeRate { get; set; }

        [LoadColumn(1)]
        public float ResidentialZones { get; set; }

        [LoadColumn(2)]
        public float CommercialZones { get; set; }

        [LoadColumn(3)]
        public float NearWater { get; set; }

        [LoadColumn(4)]
        public float ToxicWasteLevels { get; set; }

        [LoadColumn(5)]
        public float AverageRoomNumber { get; set; }

        [LoadColumn(6)]
        public float HomeAge { get; set; }

        [LoadColumn(7)]
        public float BusinessCenterDistance { get; set; }

        [LoadColumn(8)]
        public float HighwayAccess { get; set; }

        [LoadColumn(9)]
        public float TaxRate { get; set; }

        [LoadColumn(10)]
        public float StudentTeacherRatio { get; set; }

        [LoadColumn(11)]
        public float PercentPopulationBelowPoverty { get; set; }

        [LoadColumn(12)]
        [ColumnName("Label")]
        public float Price { get; set; }
    }
}
