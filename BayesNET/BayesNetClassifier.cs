using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BayesNET
{
    class BayesNetClassifier
    {
        public static List<List<string>> trainData;
        public static Dictionary<int, List<string>> trainFeatures;
        public static List<List<string>> testData;
        public static Dictionary<int, List<string>> testFeatures;
        public static Dictionary<string, int> columns;
        public static Dictionary<string, Dictionary<string, double>> CPT;//Conditional Probability Table
        public static Dictionary<string, string> bayesNetStructure;//Model's Bayes Net Structure
        public bool baseModelFlag = false;

        public BayesNetClassifier()
        {
            this.baseModelFlag = true;
        }

        public BayesNetClassifier(Dictionary<string, string> bayesianNetworkStructure)
        {
            bayesNetStructure = bayesianNetworkStructure;
        }

        public void Fit(string path, bool writeCPTtoFile)
        {
            FileOperations.Read(path, "train");// Reads train data
            if (baseModelFlag) bayesNetStructure = CreateBaseModel();// Creates base bayes structure if not given any custom structure
            DataPreprocess.CreateCPT(writeCPTtoFile);// Creates Conditional Probability Tables
        }

        public List<string> Predict(string path)
        {
            FileOperations.Read(path, "test");// Reads test data
            List<string> predictions = new List<string>();       

            for (int i = 0; i < testFeatures[0].Count; i++)
            {
                Dictionary<string, BigFloat> classScores = GetClassScores(i, false);// Gets class scores for each test sample with passZeroProbs=false parameter
                if (classScores.OrderByDescending(x => x.Value).First().Value == 0) classScores = GetClassScores(i, true);
                // If all scores is zero GetClassScores method works again with passZeroProbs=true parameter. This time, method passes all zero probabilities
                predictions.Add(classScores.OrderByDescending(x => x.Value).First().Key);// Gets highest probability class as a prediction
                classScores.Clear();
            }
            return predictions;
        }

        public Dictionary<string, BigFloat> GetClassScores(int i, bool passZeroProbs)
        {
            // Github Link: https://github.com/Osinko/BigFloat
            // This method uses BigFloat library for high precision probability calculations
            // When multiplication of probability values is performed consecutively, the value gets smaller and smaller
            List<string> uniqueClasses = trainFeatures[columns.Count - 1].Distinct().ToList();// Gets unique classes
            string classColumn = columns.Keys.Last();
            Dictionary<string, BigFloat> classScores = new Dictionary<string, BigFloat>();
            BigFloat numerator;
            BigFloat denominator = new BigFloat(0);
            bool denominatorFlag = false;
            double eps = 0.00001;// if passZeroProbs is selected, use this value as a probability value
            // CPT's consist of (part1|part2, prob_value) like strings. Each part has (rule$value) like elements. (rule$value) equals to (rule=value)
            // Example CPT element: (credit_history$all paid|own_telephone$none, class$bad)
            foreach (string uqClass in uniqueClasses)// Calculates probability values for each class
            {
                string queryP1;
                string part1, part2;
                // Creates query according to bayes formula; 
                // P(class=x|test_sample)=(P(test_sample|class=x)P(class=x))/P(test_sample)
                numerator = new BigFloat(1);
                foreach (string str in columns.Keys)
                {
                    if (str == classColumn)// P(class=x) part
                    {
                        part1 = String.Format("{0}${1}", str, uqClass);
                        part2 = "";
                    }
                    else // P(test_sample|class=x) part
                    {    // Creates query from test data infos like '(part1|part2_element1,part2_element2)'
                        part1 = String.Format("{0}${1}|", str, testFeatures[columns[str]][i]);
                        part2 = "";
                        string[] part2Split = bayesNetStructure[str].Split(',');
                        string splitter = "";
                        for (int j = 0; j < part2Split.Length; j++)
                        {
                            if (part2Split[j] == classColumn) part2 += String.Format("{0}{1}${2}", splitter, part2Split[j], uqClass);
                            else part2 += String.Format("{0}{1}${2}", splitter, part2Split[j], testFeatures[columns[part2Split[j]]][i]);
                            splitter = ",";
                        }
                    }
                    queryP1 = part1 + part2;
                    double prob = CPT[str][queryP1];// Gets query's probability values from CPT
                    if (prob == 0 && passZeroProbs) prob = eps;
                    numerator = numerator.Multiply(new BigFloat(prob));

                }

                if(!denominatorFlag)// P(test_sample) part
                {
                    denominatorFlag = true;
                    denominator = new BigFloat(0);
                    foreach (string uqInnerClass in uniqueClasses)// Same calculation process with numerator, one difference is this loop calculates each possibilities for classes
                    {
                        BigFloat tempDenominator = new BigFloat(1);
                        foreach (string str in columns.Keys)
                        {
                            if (str == classColumn)// P(class=x) part
                            {
                                part1 = String.Format("{0}${1}", str, uqInnerClass);
                                part2 = "";
                            }
                            else // P(test_sample|class=x) part
                            {    // Creates query from test data infos like '(part1|part2_element1,part2_element2)'
                                part1 = String.Format("{0}${1}|", str, testFeatures[columns[str]][i]);
                                part2 = "";
                                string[] part2Split = bayesNetStructure[str].Split(',');
                                string splitter = "";
                                for (int j = 0; j < part2Split.Length; j++)
                                {
                                    if (part2Split[j] == classColumn) part2 += String.Format("{0}{1}${2}", splitter, part2Split[j], uqInnerClass);
                                    else part2 += String.Format("{0}{1}${2}", splitter, part2Split[j], testFeatures[columns[part2Split[j]]][i]);
                                    splitter = ",";
                                }
                            }
                            queryP1 = part1 + part2;
                            double prob = CPT[str][queryP1];// Gets query's probability values from CPT
                            if (prob == 0 && passZeroProbs) prob = eps;
                            tempDenominator = tempDenominator.Multiply(new BigFloat(prob));

                        }
                        denominator = denominator.Add(tempDenominator);
                    }
                }
                
                if (denominator == 0) classScores[uqClass] = 0;
                else classScores[uqClass] = numerator.Divide(denominator);
            }

            return classScores;
        }

        public Dictionary<string, string> CreateBaseModel() // Creates base model, all nodes directly connects class node
        {                                                   // With this structure bayesNET works like classical naive bayes classifier
            Dictionary<string, string> baseModel = new Dictionary<string, string>();
            string className = columns.Keys.Last();
            foreach (string str in columns.Keys)
            {
                if (str != className) baseModel[str] = className;
                else baseModel[str] = "";
            }
            return baseModel;
        }

    }
}
