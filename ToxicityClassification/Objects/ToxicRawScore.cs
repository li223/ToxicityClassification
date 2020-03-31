using Microsoft.ML.Data;

namespace ToxicityClassification.Objects
{
    public class ToxicRawScore
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }

        [ColumnName("Score")]
        public float RawScore { get; set; }
    }
}
