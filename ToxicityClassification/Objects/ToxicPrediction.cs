using Microsoft.ML.Data;

namespace ToxicityClassification.Objects
{
    public class ToxicPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Probability")]
        public float ProbabilityOfBeingToxic { get; set; }

        [ColumnName("Score")]
        public float RawScore { get; set; }
    }
}
