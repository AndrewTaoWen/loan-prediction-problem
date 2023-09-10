import pandas as pd

test_df = pd.read_csv('./archive/test_Y3wMUE5_7gLdaTN.csv')
train_df = pd.read_csv('./archive/train_u6lujuX_CVtuZ9i.csv')

# View the first few rows of the training data
# print(train_df.head())

# View summary statistics
# print(train_df.describe())

# View data types and null values
# print(train_df.info())

# Impute missing numerical values with mean
columnsMissingVals = ["Gender", "Married", "Dependents", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Credit_History"] 

# Impute
for col in columnsMissingVals:
    if train_df[col].dtype == 'object':  # If the data type is object (likely a string)
        train_df[col].fillna(train_df[col].mode()[0], inplace=True)  # Impute with the most frequent value (mode)
    else:
        train_df[col].fillna(train_df[col].mean(), inplace=True)  # For numerical columns, impute with mean
print(train_df.info())

for col in columnsMissingVals:
    if test_df[col].dtype == 'object':  # If the data type is object (likely a string)
        test_df[col].fillna(test_df[col].mode()[0], inplace=True)  # Impute with the most frequent value (mode)
    else:
        test_df[col].fillna(test_df[col].mean(), inplace=True)  # For numerical columns, impute with mean
print(test_df.info())

all
