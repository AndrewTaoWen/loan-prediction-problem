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
# scores = cross_val_score(model, X, y, cv=5)
# print("Cross-validation scores:", scores)
# print("Average cross-validation score:", scores.mean())

val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Using RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
val_predictions_rf = rf.predict(X_val)
print("Random Forest Validation Accuracy:", accuracy_score(y_val, val_predictions_rf))

# Using GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
val_predictions_gb = gb.predict(X_val)
print("Gradient Boosting Validation Accuracy:", accuracy_score(y_val, val_predictions_gb))

# Hyperparameter Tuning on Logistic Regression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)

# Using the best model
best_model = grid.best_estimator_
val_predictions_best = best_model.predict(X_val)
print("Best Model Validation Accuracy:", accuracy_score(y_val, val_predictions_best))

# Classification report for detailed metrics
# print(classification_report(y_val, val_predictions_best))
