using Microsoft.ML.Data;

namespace ToxicityClassification.Objects
{
    public class Toxicity
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
