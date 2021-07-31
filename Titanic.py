# Importing libraries and modules
from Helpers.data_prep import *
from Helpers.eda import *
from Helpers.ML_Helpers import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Settings to display data on the console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

# I load the Titanic dataset using "load_titanic()" method
df_titanic = load_titanic()

# Checking the first five records in our dataset
df_titanic.head()

# "check_df" method returns the shape of dataset, types of variables(categorical, numeric)
# number of missing values and quantiles
check_df(df_titanic)

# "grab_col_names" method returns us the type of variables
# I will explain this method at "Read Me" part in detail
cat_cols, num_cols, cat_but_car = grab_col_names(df_titanic)

# "cat_summary" method returns the ratio of each observation for each categorical variable
for col in cat_cols:
    cat_summary(df_titanic, col)

# CHECKING FOR OUTLIERS
# We should choose between "transforming the value" or "deleting the value"
# For this dataset, I decided to "transform the outlier values" due to the insufficient number of observations
# I used "check_outliers" method to determine whether numerical variables have outliers or not
for col in num_cols:
    print(check_outlier(df_titanic, col))

# I transformed the outliers with the threshold values.
# I determined this thresholds using "Interquartile Range Method" that I will explain at "Read Me" in detail
# "replace_with_thresholds" method contains the logic of "Interquartile Range Method" also
for col in num_cols:
    replace_with_thresholds(df_titanic, col)

# CHECKING FOR MISSING VALUES

# "missing_values_table" column returns the variables with missing values, number of them and their ratio
missing_values_table(df_titanic)
#################################
#          n_miss  ratio
#Cabin        687 77.100
#Age          177 19.870
#Embarked       2  0.220

# I decided to fill the missing values with median for the "Age" variable
df_titanic['Age'].fillna(df_titanic['Age'].median(), inplace=True)

# I decided to fill the missing values with mode for the "Embarked" variable
df_titanic['Embarked'].fillna("S", inplace=True)

# I think "Ticket", "Name" and "Cabin" variables are not significant to determine whether someone survived or not
# So I decided to remove them
remove_cols = ["Ticket", "Name", "Cabin"]
df_titanic.drop(remove_cols, inplace=True, axis=1)

# FEATURE ENGINEERING

# Using correlation values, we can understand that
# "Fare, Sex, Pclass, Embarked" variables have high correlation with our target variable
# So I decided to create new features mostly using these variables

# To calculate correlation I turned categorical values into numerical ones
df_titanic['Embarked'].replace('S', 0, inplace=True)
df_titanic['Embarked'].replace('C', 1, inplace=True)
df_titanic['Embarked'].replace('Q', 2, inplace=True)
df_titanic['Sex'].replace('female', 0, inplace=True)
df_titanic['Sex'].replace('male', 1, inplace=True)
corr_matrix = df_titanic.corr()
print(corr_matrix["Survived"].sort_values(ascending=False))
# Fare           0.317
# Parch          0.082
# Embarked       0.109
# PassengerId   -0.005
# SibSp         -0.035
# Age           -0.065
# Pclass        -0.338
# Sex           -0.543

# Visualize ["Embarked", "Sex", "Pclass"] variables to see their relation with "Survived" variable
for col in ["Embarked", "Sex", "Pclass"]:
    sns.countplot(x=df_titanic[col], hue=df_titanic['Survived'])
    print(df_titanic[[col, 'Survived']].groupby([col], as_index=False).mean())
    plt.show()

# In this part, I created some new features to increase the accuracy of prediction.
def titanic_feature_engineering(df):
    # "NEW_FAMILY_SIZE" variable tries to understand whether having relatives in Titanic and number of them
    # +++has an effect on surviving or not
    # To understand this, I added up the "Sibsp(number of siblings)" and "Parch(number of parents and children)"
    # +++variables for each observation
    df["NEW_FAMILY_SIZE"] = (df["SibSp"]) + (df["Parch"]) + 1

    # Also using the same logic with the above feature, I created "New_Is_Alone" feature
    # Using this feature, I plan to understand the effect of being lonely in Titanic on surviving
    # 0 => Not Alone        1 => Alone
    df.loc[df['SibSp'] + df['Parch'] == 0, "New_Is_Alone"] = "1"
    df.loc[df['SibSp'] + df['Parch'] > 0, "New_Is_Alone"] = "0"

    # I created a new feature to determine ratio of survival for wealthier female.
    # 0 => Rich     1 => Not Rich
    df_titanic.loc[(df_titanic["Embarked"] == 0) & (df_titanic["Sex"] == 0) & (df_titanic["Pclass"] == 1),
                   "New_RichPeople"] = "1"
    df_titanic.loc[(df_titanic["Embarked"] != 0) | (df_titanic["Sex"] != 0) | (df_titanic["Pclass"] != 1),
                   "New_RichPeople"] = "0"

    # 1 => First Class Female   2 => Second Class Female    3 => Third Class Female 4=> Males
    df_titanic.loc[(df_titanic["Pclass"] == 1) & (df_titanic["Sex"] == 0), "New_Female_Pclass"] = "1"
    df_titanic.loc[(df_titanic["Pclass"] == 2) & (df_titanic["Sex"] == 0), "New_Female_Pclass"] = "2"
    df_titanic.loc[(df_titanic["Pclass"] == 3) & (df_titanic["Sex"] == 0), "New_Female_Pclass"] = "3"
    df_titanic.loc[(df_titanic["Sex"] == 1), "New_Female_Pclass"] = "4"
titanic_feature_engineering(df_titanic)

# After creating new features I re-implement all the EDA operations again
check_df(df_titanic)
cat_cols, num_cols, cat_but_car = grab_col_names(df_titanic)
num_cols = [col for col in num_cols if "PassengerId" not in col]
for col in cat_cols:
    cat_summary(df_titanic, col)


#############################################
# One_Hot_Encoding
#############################################

# Datasets consists of both "categorical" and "numerical" values most of the time
# But ML algorithms waits for numerical values to process them
# One Hot Encoding is a way to deal with this problem
# On the next line there is a list comprehension to filter columns for OHE
ohe_cols = [col for col in df_titanic.columns if 10 >= df_titanic[col].nunique() > 2]
df_titanic = one_hot_encoder(df_titanic, ohe_cols)

# After OHE, we have many new variables, so we need to split them as categorical and numerical again
cat_cols, num_cols, cat_but_car = grab_col_names(df_titanic)
num_cols = [col for col in num_cols if "PassengerId" not in col]

#############################################
# Drop Useless Columns
#############################################

# "rare_analyser" and "cols_to_drop" gives us the new variables which have low frequency
# Because of the low frequency I decided to drop them
rare_analyser(df_titanic, "Survived", cat_cols)
cols_to_drop = [col for col in df_titanic.columns if df_titanic[col].nunique() == 2 and
                (df_titanic[col].value_counts() / len(df_titanic) < 0.05).any(axis=None)]

df_titanic.drop(cols_to_drop, axis=1, inplace=True)
df_titanic.drop(columns=['PassengerId'], axis=1, inplace=True)

# Feature Scaling With Robust Scaler

# We can use scaling to reduce the effect of massive numerical values
scaler = RobustScaler()
df_titanic[num_cols] = scaler.fit_transform(df_titanic[num_cols])

##########################
# Fitting Model
##########################

# Dependent Variable
y = df_titanic["Survived"]
# Independent Variables
X = df_titanic.drop(["Survived"], axis=1)


# I split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33, random_state=1)

# Here are some ML algorithms that I will use to predict my target variable
log_model = LogisticRegression().fit(X_train, y_train)
rfm = RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1, random_state=101,
                            max_features=None, min_samples_leaf=30).fit(X_train, y_train)
dt = DecisionTreeClassifier(max_depth=10, random_state=101,
                            max_features=None, min_samples_leaf=15).fit(X_train, y_train)
nb = GaussianNB().fit(X_train, y_train)

model_list = [log_model, rfm, dt, nb]

# I wrote a method to display "Accuracy, Precision, Recall and F1 Scores" which I will explain in detail at "Read Me"
# Also I used ROC_AUC Score for prediction
def model_processor():
    for model in model_list:
        y_pred = model.predict(X_train)

        ##########################
        # Evaluation of Predictions
        ##########################

        # Train Accuracy
        print(f"Train Accuracy for {model}: {accuracy_score(y_train, y_pred)}")

        # Test Accuracy

        # For AUC Score  y_prob
        y_prob = model.predict_proba(X_test)[:, 1]

        # For the other metrics y_pred
        y_pred = model.predict(X_test)

        # ACCURACY
        print(f"Test Accuracy for {model}: {accuracy_score(y_test, y_pred)}")

        # PRECISION
        print(f"Precision Score for {model}: {precision_score(y_test, y_pred)}")

        # RECALL
        print(f"Recall Score for {model}: {recall_score(y_test, y_pred)}")

        # F1
        print(f"F1 Score for {model}: {f1_score(y_test, y_pred)}")

        # ROC CURVE
        plot_ROC_curve(model, X_test, y_test)

        # AUC
        print(f"AUC Score for {model}: {roc_auc_score(y_test, y_prob)}")

        # Classification report
        print(f"CLASSIFICATION REPORT FOR {model}")
        print(classification_report(y_test, y_pred))

    for models in [rfm, dt]:
        print(f"IMPORTANCE PLOT FOR {models}")
        plot_importance(models, X, 10) #final modelinin değişken önem düzeylerine bakılır.

model_processor()