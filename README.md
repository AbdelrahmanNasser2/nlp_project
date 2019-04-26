# NLP project

# Sentiment Analysis and Trend Detection

This is sentiment analysis project use IMDb movie review dataset 

# The Results of Machine Learning Experiments :
we do looping on The hyperparameters "C" values which adjusts the regularization 
C_values = [0.01, 0.05, 0.25, 0.5, 1]


- First Experiment : Baseline Code With Logistic Regression

      The accuracy of the 5 C : [0.86736, 0.87408, 0.87264, 0.87088, 0.8704]
      The Highest C : 0.05
      Final Accuracy: 0.88004

      Precision: 0.879
      Recall: 0.885
      F1: 0.882

- Second Experiment : Bi-Grams With Logistic Regression

      The accuracy of the 5 C : [0.8888, 0.8936, 0.89488, 0.89568, 0.89584]
      The Highest C : 1
      Final Accuracy: 0.89788

      Precision: 0.893
      Recall: 0.903
      F1: 0.898

- Third Experiment : Word Counts With Logistic Regression

      The accuracy of the 5 C : [0.87632, 0.88304, 0.88432, 0.8824, 0.88304]
      The Highest C : 0.25
      Final Accuracy: 0.87232

      Precision: 0.883
      Recall: 0.881
      F1: 0.882

- Forth Experiment : TF-IDF With Logistic Regression

      The accuracy of the 5 C : [0.7984, 0.83472, 0.8728, 0.88, 0.88704]
      The Highest C : 1
      Final Accuracy: 0.88176

      Precision: 0.883
      Recall: 0.882
      F1: 0.882

- Fifth Experiment : Bi-Gram With Support Vector Machine Model

      The accuracy of the 5 C : [0.89248, 0.8904, 0.8904, 0.89072, 0.89104]
      The Highest C : 0.01
      Final Accuracy: 0.89708

      Precision: 0.892
      Recall: 0.903
      F1: 0.898

# Conclusion :
the Bi-Grams with Logistic Regression get the best accuracy and best F1-Score.
