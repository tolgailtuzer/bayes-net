using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace BayesNET
{
    class DataPreprocess
    {
        public static void CreateCPT(bool writeFile)
        {   // Creates every possible query with training data's unique class elements
            BayesNetClassifier.CPT = new Dictionary<string, Dictionary<string, double>>();
            foreach (string name in BayesNetClassifier.columns.Keys)
            {
                if (BayesNetClassifier.bayesNetStructure[name] == "")// Root element case
                {
                    List<string> queryList = BayesNetClassifier.trainFeatures[BayesNetClassifier.columns[name]].Distinct().ToList();// Gets unique elements of current column
                    for (int i = 0; i < queryList.Count; i++)
                    {
                        queryList[i] = String.Format("{0}${1}", name, queryList[i]);// Creates every possible query like (name$bad). (name$bad) equals to (name=bad)
                    }
                    queryList.Sort();// List is sorting for a common notation
                    BayesNetClassifier.CPT[name] = CalculateProbability(name, queryList, writeFile);// Calculates probabilities of given table
                }
                else // has child element case
                {

                    List<string> queryList = Cartesian(name, BayesNetClassifier.bayesNetStructure[name]);// Creates every possible query list with using cartesian product
                    BayesNetClassifier.CPT[name] = CalculateProbability(name, queryList, writeFile);// Calculates probabilities of given table
                }
            }
        }

        public static Dictionary<string, double> CalculateProbability(string name, List<string> queries, bool writeFile)
        {
            string splitter = "";
            Dictionary<string, double> queryProbs = new Dictionary<string, double>();
            Dictionary<string, string> cptTable = new Dictionary<string, string>();
            foreach (string q in queries)
            {   // Queries -> (left_side_element|right_side_element1, right_side_element2, ...) split into parts and finds count in training data for CPT
                string part1, part2;
                string[] querySplit = q.Split('|');
                if (querySplit.Length == 1)// Root part case
                {
                    part1 = querySplit[0];
                    part2 = "";
                }
                else // (part1 -> left_side_element, right_side_element1, right_side_element2, ...) (part2 -> right_side_element1,right_side_element2, ...)
                {    // Each element consist of equation like (a$b) equals to (a=b)
                    part1 = querySplit[0] + ',' + querySplit[1];
                    part2 = querySplit[1];
                }
                int part1Counter = 0, part2Counter = 0;
                for (int i = 0; i < BayesNetClassifier.trainFeatures[0].Count; i++)// Finds how many data points provides the given equation
                {
                    Boolean part1Status = true;
                    string[] part1Split = part1.Split(',');
                    for (int j = 0; j < part1Split.Length; j++)
                    {
                        string[] query = part1Split[j].Split('$');
                        part1Status &= (BayesNetClassifier.trainFeatures[BayesNetClassifier.columns[query[0]]][i] == query[1]);
                    }
                    if (part1Status) part1Counter++;

                    Boolean part2Status = true;
                    if (part2 != "")
                    {
                        string[] part2Split = part2.Split(',');
                        for (int j = 0; j < part2Split.Length; j++)
                        {
                            string[] query = part2Split[j].Split('$');
                            part2Status &= (BayesNetClassifier.trainFeatures[BayesNetClassifier.columns[query[0]]][i] == query[1]);
                        }
                    }
                    if (part2Status) part2Counter++;
                }

                double prob = part1Counter / (double)(part2Counter);
                queryProbs[q] = prob;

                // Prepares case probability dictionary for log file
                if (part2=="")// Root case
                {
                    if (!cptTable.ContainsKey(""))
                    {
                        cptTable[""] = "";
                        splitter = "";
                    }
                    cptTable[""]+= (splitter + prob);
                    splitter = ",";

                }
                else// The other cases
                {
                    string[] rseSplit = part2.Split(',');
                    string value = "";
                    splitter = "";
                    for (int j = 0; j < rseSplit.Length; j++)
                    {
                        string[] query = rseSplit[j].Split('$');
                        value += (splitter + query[1]);
                        splitter = ",";
                    }
                    if (!cptTable.ContainsKey(value))
                    {
                        cptTable[value] = "";
                    }
                    cptTable[value] += ("," + prob);
                }
                
            }
            if(writeFile)
            {
                StreamWriter sw = new StreamWriter(String.Format("{0}.csv", name));
                string[] rightSide = BayesNetClassifier.bayesNetStructure[name].Split(',');
                splitter = "";
                foreach (string rse in rightSide)// Writes parents of given table to file
                {
                    sw.Write(splitter + rse);
                    if (rse != "") splitter = ",";
                }
                List<string> columnNames = BayesNetClassifier.trainFeatures[BayesNetClassifier.columns[name]].Distinct().ToList();
                columnNames.Sort();
                foreach (string lse in columnNames)// Writes current table's elements to file
                {
                    sw.Write(splitter + lse);
                    splitter = ",";
                }
                sw.Write("\n");

                foreach (KeyValuePair<string, string> elements in cptTable)// Writes all possible cases to file
                {
                    sw.WriteLine(elements.Key + elements.Value);
                }
                sw.Close();
            }
            return queryProbs;

        }

        public static List<string> Cartesian(string leftSide, string rightSide)
        {   // String format for CPT's is (part1|part2_element1, part2_element2, ...)
            List<List<string>> sets = new List<List<string>>();
            string query = leftSide + ',' + rightSide;
            string[] querySplit = query.Split(',');
            foreach (string q in querySplit)// Creates sets from columns unique elements
            {
                List<string> tempList = BayesNetClassifier.trainFeatures[BayesNetClassifier.columns[q]].Distinct().ToList();
                for (int i = 0; i < tempList.Count; i++)
                {
                    tempList[i] = String.Format("{0}${1}", q, tempList[i]);
                }
                tempList.Sort();
                sets.Add(tempList);
            }
            List<string> result = sets[0];
            for (int i = 1; i < sets.Count; i++)// Use sets for cartesian product
            {
                result = CartesianProduct(result, sets[i]);// Applies cartesian product process iteratively
            }

            Regex rgx = new Regex(",");
            // Replaces first comma with '|' character and gets CPT's string format (part1|part2_element1, part2_element2, ...)
            for (int i = 0; i < result.Count; i++)
            {
                result[i] = rgx.Replace(result[i], "|", 1);
            }
            return result;
        }

        public static List<string> CartesianProduct(List<string> list1, List<string> list2)
        {
            List<string> result = new List<string>();
            for (int i = 0; i < list1.Count; i++)
            {
                for (int j = 0; j < list2.Count; j++)
                {
                    result.Add(list1[i] + ',' + list2[j]);
                }
            }
            return result;
        }
    }
}
