import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split  # Importing train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Load the data
train_df = pd.read_csv('./archive/train_u6lujuX_CVtuZ9i.csv')
train_df['Loan_Status'] = train_df['Loan_Status'].map({'Y': 1, 'N': 0})  # Encoding target

columnsMissingVals = ["Gender", "Married", "Dependents", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

for col in columnsMissingVals:
    if train_df[col].dtype == 'object':
        train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    else:
        train_df[col].fillna(train_df[col].mean(), inplace=True)

train_df = pd.get_dummies(train_df, columns=['Gender', 'Married', 'Dependents', 'Self_Employed', 'Education', 'Property_Area'], drop_first=True)
train_df *= 1

'''
for column in train_df:
    if column != 'Loan_Status':
        plot = train_df.plot(x = column, y = 'Loan_Status', kind = 'scatter')
        plt.show()
'''


# Feature scaling
#scaler = StandardScaler()
#train_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']] = scaler.fit_transform(train_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']])

# Drop Loan_ID
train_df.drop('Loan_ID', axis=1, inplace=True)

# Separate features and target variable from train_df
#X = train_df.drop('Loan_Status', axis=1)
subset1 = train_df[train_df['Credit_History'] == 1]
#X = train_df.drop(['Loan_Status', 'Gender_Male', 'Married_Yes'], axis=1)
X1 = subset1.drop(['Loan_Status', 'Gender_Male', 'Married_Yes'], axis=1)
y1 = subset1['Loan_Status']

# Split the data: 70% training and 30% test
X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y1, test_size=0.3, random_state=42)

# Fit the model
model = LogisticRegression(max_iter=2000, C=1.0, solver='newton-cg')
#rfe = RFE(estimator=model, n_features_to_select=10)
fit = model.fit(X_train1, y_train1)

# Make predictions on the validation set
val_predictions1 = fit.predict(X_val1)

subset2 = train_df[train_df['Credit_History'] == 0]
#X = train_df.drop(['Loan_Status', 'Gender_Male', 'Married_Yes'], axis=1)
X2 = subset2.drop(['Loan_Status', 'Gender_Male', 'Married_Yes'], axis=1)
y2 = subset2['Loan_Status']

# Split the data: 70% training and 30% test
X_train2, X_val2, y_train2, y_val2 = train_test_split(X2, y2, test_size=0.3, random_state=42)

# Fit the model
model = LogisticRegression(max_iter=2000, C=1.0, solver='newton-cg')
#rfe = RFE(estimator=model, n_features_to_select=10)
fit = model.fit(X_train2, y_train2)

# Make predictions on the validation set
val_predictions2 = fit.predict(X_val2)

val_accuracy = (accuracy_score(y_val1, val_predictions1)*len(y_val1) + accuracy_score(y_val2, val_predictions2)*len(y_val2))/(len(y_val1) + len(y_val2))
print(len(y1)/len(train_df.index))
print(len(y2)/len(train_df.index))
print(accuracy_score(y_val1, val_predictions1))
print(accuracy_score(y_val2, val_predictions2))
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")