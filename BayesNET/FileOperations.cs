using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BayesNET
{
    class FileOperations
    {
        public static void Read(string path, string type)
        {
            CultureInfo.DefaultThreadCurrentCulture = new CultureInfo("en-US");//for separating decimal numbers with dots
            StreamReader sr = new StreamReader(path);
            if (type == "train")
            {
                BayesNetClassifier.trainData = new List<List<string>>();
                BayesNetClassifier.trainFeatures = new Dictionary<int, List<string>>();
            }
            else
            {
                BayesNetClassifier.testData = new List<List<string>>();
                BayesNetClassifier.testFeatures = new Dictionary<int, List<string>>();
            }
            int i = 0, j;
            string[] temp_columns = sr.ReadLine().Split(',');

            if (type == "train")
            {
                BayesNetClassifier.columns = new Dictionary<string, int>();
                for (int cc = 0; cc < temp_columns.Length; cc++)
                {
                    BayesNetClassifier.columns[temp_columns[cc]] = cc;
                }
            }

            while (!sr.EndOfStream)
            {
                string[] line = sr.ReadLine().Split(',');
                List<string> temp_data = new List<string>();
                for (j = 0; j < line.Length; j++)
                {
                    temp_data.Add(line[j]);
                    if (i == 0)//In the beginning creates lists for features
                    {
                        List<string> temp_features = new List<string>();
                        temp_features.Add(line[j]);
                        if (type == "train") BayesNetClassifier.trainFeatures.Add(j, temp_features);
                        else BayesNetClassifier.testFeatures.Add(j, temp_features);
                    }
                    else//If there is a list for current feature, value adds to this list
                    {
                        if (type == "train") BayesNetClassifier.trainFeatures[j].Add(line[j]);
                        else BayesNetClassifier.testFeatures[j].Add(line[j]);
                    }
                }
                if (type == "train") BayesNetClassifier.trainData.Add(temp_data);
                else BayesNetClassifier.testData.Add(temp_data);
                i++;
            }
            sr.Close();
        }
        
    }
}
