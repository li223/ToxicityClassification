using Microsoft.ML;
using Microsoft.ML.Trainers;
using ToxicityClassification.Objects;
using System;
using System.IO;
using System.Linq;
using System.Threading;

namespace ToxicityClassification
{
    class Program
    {
        private static readonly MLContext mlctx = new MLContext(1);
        static void Main(string[] _)
        {
            TrainBinaryModel();
            while (true)
            {
                Console.Write("\nEnter Text to Test: ");
                var txt = Console.ReadLine();
                TestModel(txt);
            }
        }

        public static void TrainBinaryModel()
        {
            var asm = new FileInfo(typeof(Program).Assembly.Location);
            string rootdir = asm.Directory.FullName;

            //Config - Convert text data into format of type Sentiment
            IDataView dataView = mlctx.Data.LoadFromTextFile<Toxicity>($"{rootdir}/Data/TestData.tsv", hasHeader: false);

            //Split test data set into training and testing data sets
            var trainTestData = mlctx.Data.TrainTestSplit(dataView, 0.4);
            var trainingData = trainTestData.TrainSet;
            var testingData = trainTestData.TestSet;

            var aasdas = dataView.Preview();

            //Transform text into usable binary set
            var dataPipline = mlctx.Transforms.Text.FeaturizeText("Features", "Text");

            //Set the algorithm and config model builder
            var trainer = mlctx.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", new SmoothedHingeLoss(), 5F, true, numberOfIterations: 5000).AppendCacheCheckpoint(mlctx);
            var trainingPipline = dataPipline.Append(trainer);

            int elapsedsecs = 0;
            var timer = new Timer((_) =>
            {
                elapsedsecs++;
                Console.Clear();
                Console.WriteLine($"Training Model -> Elapsed Time: {elapsedsecs}s");
            }, null, 0, 1000);

            //Fit model to training data
            var trainedModel = trainingPipline.Fit(trainingData);
            timer.Dispose();

            //Evaluate and show results
            Console.WriteLine("Testing and Evaluating Model");
            var pred = trainedModel.Transform(testingData);

            var sentment = mlctx.BinaryClassification.EvaluateNonCalibrated(pred);

            var caliest = mlctx.BinaryClassification.Calibrators.Platt();
            var calitrans = caliest.Fit(pred);
            var finaldatatrans = calitrans.Transform(pred);

            //Save Model
            Console.WriteLine("Saving Model");
            mlctx.Model.Save(trainedModel, trainingData.Schema, $"{rootdir}/Model/Model.zip");

            //Sample sentence prediction
            var sample = new Toxicity { Text = "Hello, how are you today?" };

            var predengine = mlctx.Model.CreatePredictionEngine<Toxicity, ToxicRawScore>(trainedModel);
            var results = predengine.Predict(sample);

            var predent = mlctx.Model.CreatePredictionEngine<ToxicRawScore, ToxicPrediction>(calitrans);
            var rest = predent.Predict(results);

            mlctx.Model.Save(calitrans, predent.OutputSchema, $"{rootdir}/Model/CaliModel.zip");

            Console.WriteLine($"Metrics -> Accuracy: {sentment.Accuracy} || Negative Precision: {sentment.NegativePrecision} || Positive Precision: {sentment.PositivePrecision}");
            Console.WriteLine($"\n\nPrediction Results: Text: {sample.Text} || Prediction: {(rest.Prediction ? "Toxic" : "Not Toxic")} || Probility: {rest.ProbabilityOfBeingToxic}");
        }

        public static void TestModel(string text)
        {
            var asm = new FileInfo(typeof(Program).Assembly.Location);
            string rootdir = asm.Directory.FullName;

            var sentfs = new FileStream(path: $"{rootdir}/Model/Model.zip", mode: FileMode.Open);
            var califs = new FileStream(path: $"{rootdir}/Model/CaliModel.zip", mode: FileMode.Open);
            
            var sentpipe = mlctx.Model.Load(sentfs, out _);
            sentfs.Close();
            sentfs.Dispose();
            
            var calipipe = mlctx.Model.Load(califs, out _);
            califs.Close();
            califs.Dispose();

            var sentpredengine = mlctx.Model.CreatePredictionEngine<Toxicity, ToxicRawScore>(sentpipe);
            var sample = new Toxicity() { Text = text };
            var results = sentpredengine.Predict(sample);

            var calipredengine = mlctx.Model.CreatePredictionEngine<ToxicRawScore, ToxicPrediction>(calipipe);
            var calires = calipredengine.Predict(results);

            Console.WriteLine($"\n\nPrediction Results: Text: {sample.Text} || Prediction: {(calires.Prediction ? "Toxic": "Not Toxic")}\n\nScore: {calires.RawScore} || Probability: {calires.ProbabilityOfBeingToxic}");
        }
    }
}