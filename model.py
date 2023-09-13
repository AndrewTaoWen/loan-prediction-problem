import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split  # Importing train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# Load the data
train_df = pd.read_csv('./archive/train_u6lujuX_CVtuZ9i.csv')
train_df['Loan_Status'] = train_df['Loan_Status'].map({'Y': 1, 'N': 0})  # Encoding target

# Impute missing values and preprocess
columnsMissingVals = ["Gender", "Married", "Dependents", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

for col in columnsMissingVals:
    if train_df[col].dtype == 'object':
        train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    else:
        train_df[col].fillna(train_df[col].mean(), inplace=True)

# One-hot encode categorical variables
train_df = pd.get_dummies(train_df, columns=['Gender', 'Married', 'Dependents', 'Self_Employed', 'Education', 'Property_Area'], drop_first=True)

# Feature scaling
scaler = StandardScaler()
train_df[['LoanAmount', 'Loan_Amount_Term']] = scaler.fit_transform(train_df[['LoanAmount', 'Loan_Amount_Term']])

# Drop Loan_ID
train_df.drop(['Loan_ID'], axis=1, inplace=True)

# Separate features and target variable from train_df
X = train_df.drop('Loan_Status', axis=1)
y = train_df['Loan_Status']

# Split the data: 70% training and 30% test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
model = LogisticRegression(max_iter=2000, C=1.0, solver='newton-cg')
rfe = RFE(estimator=model, n_features_to_select=10)
fit = rfe.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = fit.predict(X_val)

# Evaluate the model on the validation set
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())

val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
