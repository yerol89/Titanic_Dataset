# Titanic_Dataset

Hello Everyone,\
In this repository, I am supposed to develop a model to predict whether people could survive or not using the given data about each passenger.\
For this dataset, the variable **Survived** is a categorical variabl. So, this is problem is a **CLASSIFICATION PROBLEM**\
In this ReadMe part, I will try to explain each part of my source code and give some detail about the theory behind each part.

## 1. EXPLORATORY DATA ANALYSIS
![Exploratory Data Analysis](https://media-exp1.licdn.com/dms/image/C4E12AQHlhF7SooEmDA/article-cover_image-shrink_600_2000/0/1602837697395?e=1632960000&v=beta&t=PStG3FAslTNIMW4_W2D4C3kUG5qGHFbs0nfNbIyL8uI)

|  Variable Name |        Definition    |            Key           |
| -------------- | ---------------------| -------------------------|
|   Survived     |    Survived or not   |     0 = No, 1 = Yes      |
|   Pclass       |      Ticket Class    | 1 = 1st, 2 = 2nd, 3 = 3rd|
|   Name         |   Name of passenger  |             -            |
|   Sex          |   Sex of passenger   |       male, female       |
|   Age          |      Age in years    |             -            |
|   SibSp        |# of siblings/ spouses|             -            |
|   Parch        |# of parents/ children|             -            |
|   Ticket       |      Ticket number   |             -            |
|   Fare         |      Passenger Fare  |             -            |
|   Cabin        |      Cabin nunber    |             -            |
|   Embarked     |  Port of embarkation |C = Cherbourg, Q = Queenstown, S = Southampton|

At line 31 we have a method named **check_df** and this method returns the shape of the dataframe, types of the variables, sum of missing values for each variable and quantile values.
At line 35 we have **grab_col_names** method. Under **Helpers** folder we have **eda.py** file. Inside this python file you can see the details about this method inside a docstring.

`check_df(df_titanic)`
- - -
This dataset has 891 observations with 12 variables.
**Age, cabin and Embarked** variables have missing values.

|  Variable Name |   Variable Types | Missing Values |
| -------------- | ---------------- | ---------------|
|   PassengerId  |      int64       |       0        |
|   Survived     |      int64       |       0        |
|   Pclass       |      int64       |       0        |
|   Name         |      object      |       0        |
|   Sex          |      object      |       0        |
|   Age          |      float64     |     177        |
|   SibSp        |      int64       |       0        |
|   Parch        |      int64       |       0        |
|   Ticket       |      object      |       0        |
|   Fare         |      float64     |       0        |
|   Cabin        |      object      |     687        |
|   Embarked     |      object      |       2        |

## 2. DATA PRE_PROCESSING
   
   ### Dealing With Outliers
   If `check_outlier(df_titanic, col)` method returns for a variable it means that there exist an outlier value in this variable.\
   Then I use `replace_with_thresholds(df_titanic, col)` method to transform this outliers with threshold values.\
   But how should we determine these thresholds?\
   I use **Interquartile Range Method** for this. You can find an excellent explanation on this link [IQR](https://online.stat.psu.edu/stat200/lesson/3/3.2)
   
   ### Dealing With Missing Values
   `missing_values_table(df_titanic)` method gave me the output below.\
   I generally fill the missing values with the median or mode values but it depends on the variable and the dataset of course.
   
|  Variable Name |   Number of Missing Values | Ratio |
| -------------- | ---------------- | ---------------|
|   Cabin  |      687       |       77.1        |
|   Age     |      177       |       19.87        |
|   Embarked       |      2       |       0.22        |

  ### Feature Engineering
  `
  corr_matrix = df_titanic.corr()
  print(corr_matrix["Survived"].sort_values(ascending=False))
  `\
  The code snippet given above, returns us the table seen below.\ 
  Using the values in this table we can easily understand that **"Fare, Sex, Pclass, Embarked"** have high correlation with our target variable "Survived".\
  So I used these variables to create new features.
  
|  Variable Name |   Correlation Value |
| -------------- | --------------------| 
|   Fare  |      0.317       | 
|   Embarked     |      0.109       | 
|   Parch       |      0.082       |
|   SibSp  |      -0.035       | 
|   Age     |      -0.065       | 
|   Pclass      |      -0.338       |
|   Sex  |      -0.543       | 

  ### Encoding
  
  Many datasets have non-numerical variables. For example in this dataset we have **Embarked** variable with **S, C and Q** values. Although some ML algorithms handle this kind
  of categorical variables, many of them waits for numerical values to fit the model.\
  So one of the most important challenges is turning categorical variables into numerical ones. To achieve this, I used **One Hot Encoder(OHE)** in this repository.\
  **OHE** converts each value into a new variable and assigns 0 or 1 as variable values.\
  In other words, **OHE** map each label to a binary vector
  
## 3. MODEL FITTING
I split my dataset into two parts as test and train sets.\
Then used train set to fit the model.\
I fit several ML algorithms for comparison in this repository.\
**How can we understand or measure our success level of prediction?**\
  ### Performance Measures
  To understand performance measures better we should first talk about the **Confusion Matrix** 
      
  ![Confusion Matrix](https://miro.medium.com/max/1000/1*fxiTNIgOyvAombPJx5KGeA.png)
  True positive and true negatives are the correctly predicted observations and shown in green. 
  We want to reduce the value for false positives and false negatives and they shown in red color.

  **True Positives (TP)** - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes. 
  For example, if actual class value says that this passenger survived and predicted class tells you the same, this can be considered as a TP value.    
  
  **True Negatives (TN)** - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no. 
  For example, if actual class says this passenger did not survive and predicted class tells also the same thing, this can be considered as TN.

  False positives and false negatives, these values occur when your actual class says the opposite of the predicted class.

  **False Positives (FP)** – When actual class is no and predicted class is yes. 
  For example, if actual class says this passenger did not survive but predicted class tells you that this passenger survives this is a FP.\ 
  In other words, actually not positive(not survived) but predicted as positive(survived).
  
  **False Negatives (FN)** – When actual class is yes but predicted class is no. 
  For example, if actual class value says that this passenger survived and predicted class tells you that passenger died.\
  In other words, actually positive(survived) but predicted as not positive(not survived).

  Once you understand these four parameters then we can calculate Accuracy, Precision, Recall and F1 score.

  **Accuracy -** Accuracy is a performance measure and it is a ratio of correctly predicted observation to the total observations. 
  One may think that, if we have high accuracy then our model is best. Yes, accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same. Therefore, you have to look at other parameters to evaluate the performance of your model.

  **Accuracy = TP+TN/TP+FP+FN+TN**

  **Precision -** Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. 
  The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate.
  In other words, what is the percentage of the people really survived, to all the people predicted as survived.
  **Precision = TP/TP+FP**

  **Recall (Sensitivity) -** Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. 
  The question recall answers is: Of all the passengers that truly survived, how many did we label?
  In other words, what is our model's success level of labeling truly survived people out of all really survived ones ? 
  **Recall = TP/TP+FN**

  **F1 score -** F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.

  **F1 Score = 2*(Recall * Precision) / (Recall + Precision)**
  
  Even after all these metrics we may need some other methods to evaluate our model success. Especially for the classification problems like our dataset, **ROC/AUC** is a popular metric also.
  I used this metric to measure the performance of my model. If you interested in these metrics and their comparisons, I strongly advice you to read this article on the [link](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc).
  
  
  
  
  












