# Bayesian Network
## About
A Bayesian network or belief network is a probabilistic graphical model (a type of statistical model) that represents a set of variables and their conditional dependencies with a directed acyclic graph (DAG). 
Bayesian networks are ideal for taking an event that occurred and predicting the likelihood that any one of several possible known causes was the contributing factor.
For example, a Bayesian network could represent the probabilistic relationships between diseases and symptoms. Given symptoms, the network can be used to compute the probabilities of the presence of various diseases. 

## Input Structure
- Input format must be CSV file.
- Model's fit method takes train file path as an argument. Train file must have label columns as a last column.
- Model's predict method takes test file path as an argument.

## About Example Dataset
- Example dataset is simplified version of <a href="https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)"> German Credit Data<a>.

<p align="center">
  <img src="/Images/bayesnetstructure.jpg"/>
  Example Bayesian Network
  <br>
  <img src="/Images/CPT.png"/>
  Example CPT(Conditional Probability Table)
</p>


## Implementation Details
- In code you can specify custom network structure. If there is no structure for algorithm, it will work like Naive Bayes. 
    - Example
    ```csharp
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
    ```
- If the writeCPTtoFile parameter is true, program prints the generated CPT's into csv files.
    ```csharp
        model.Fit(trainPath, true);
    ```

