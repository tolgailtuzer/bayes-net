using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BayesNET
{
    class Program
    {
        static void Main(string[] args)
        {
            Dictionary<string, string> customBayesNetStructure = new Dictionary<string, string>
            {
                {"class", ""},
                {"property_magnitude", "class"},
                {"housing", "property_magnitude,class"},
                {"purpose", "housing,class"},
                {"personal_status", "housing,class"},
                {"job", "property_magnitude,class"},
                {"employment", "job,class"},
                {"own_telephone", "job,class"},
                {"credit_history", "own_telephone,class"}
            };

            string trainPath = @"..\..\..\Datasets\Proje1-Train.txt";
            string testPath = @"..\..\..\Datasets\Proje1-Test.txt";

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            // When there is no custom structure for model, it works like a classical bayes classifier
            BayesNetClassifier model = new BayesNetClassifier(customBayesNetStructure);
            
            model.Fit(trainPath, false);// Second boolean parameter is a choice of writing CPT's to file

            List<string> predictions = model.Predict(testPath);

            // Classification Report
            EvaluationMetrics.GetModelReport(predictions, "bad", "good", stopWatch.ElapsedMilliseconds);

        }
    }
}
