using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BayesNET
{
    class EvaluationMetrics
    {
        public static Dictionary<string, int> GetConfusionMatrix(List<string> testTrue, List<string> testPredict, string positiveLabel, string negativeLabel)
        {
            Dictionary<string, int> confMatrix = new Dictionary<string, int>();
            confMatrix["TP"] = 0;
            confMatrix["FP"] = 0;
            confMatrix["TN"] = 0;
            confMatrix["FN"] = 0;
            for (int i = 0; i < testTrue.Count; i++)
            {
                if (testTrue[i] == testPredict[i] && testTrue[i] == positiveLabel) confMatrix["TP"]++;
                if (testTrue[i] != testPredict[i] && testPredict[i] == positiveLabel) confMatrix["FP"]++;
                if (testTrue[i] == testPredict[i] && testTrue[i] == negativeLabel) confMatrix["TN"]++;
                if (testTrue[i] != testPredict[i] && testPredict[i] == negativeLabel) confMatrix["FN"]++;
            }
            return confMatrix;
        }

        public static Dictionary<string, double> GetMetrics(Dictionary<string, int> confusionMatrix)
        {
            Dictionary<string, double> metrics = new Dictionary<string, double>();
            metrics["TP_RATE"] = confusionMatrix["TP"] / (double)(confusionMatrix["TP"] + confusionMatrix["FN"]);
            metrics["TN_RATE"] = confusionMatrix["TN"] / (double)(confusionMatrix["TN"] + confusionMatrix["FP"]);
            metrics["ACCURACY"] = (confusionMatrix["TP"]+ confusionMatrix["TN"]) / (double)(confusionMatrix["TP"] + confusionMatrix["FP"]+ confusionMatrix["TN"]+ confusionMatrix["FN"]);
            return metrics;
        }

        public static void GetModelReport(List<string> predictions, string positiveLabel, string negativeLabel, long elapsedTime)
        {
            Dictionary<string, int> conf = EvaluationMetrics.GetConfusionMatrix(BayesNetClassifier.testFeatures[BayesNetClassifier.columns.Count - 1], predictions, "bad", "good");

            Dictionary<string, double> metrics = EvaluationMetrics.GetMetrics(conf);

            Console.WriteLine(String.Format("Elapsed Time: {0} ms", elapsedTime));
            Console.WriteLine("Confusion Matrix");
            Console.WriteLine("--------------");
            Console.WriteLine(String.Format("   {0,-3}    {1,-3}\n   {2,-3}    {3,-3}", conf["TP"], conf["FN"], conf["FP"], conf["TN"]));
            Console.WriteLine("--------------");
            Console.WriteLine(String.Format("TP_Rate: {0}\nTN_Rate: {1}\nAccuracy: {2}", metrics["TP_RATE"], metrics["TN_RATE"], metrics["ACCURACY"]));
            Console.Read();
        }
    }
}
