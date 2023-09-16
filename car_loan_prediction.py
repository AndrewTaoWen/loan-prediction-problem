import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('./car_loan_data/train.csv')

# Identify the columns of object data type
object_cols = df.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
le = LabelEncoder()

# Loop over object columns to transform them
for col in object_cols:
    df[col].fillna('Unknown', inplace=True)  # Handle missing values
    df[col] = le.fit_transform(df[col])  # Convert to numerical labels

# Handle missing values (Impute or drop based on your specific use-case)
df['Employment.Type'].fillna('Unknown', inplace=True)

# Drop the original date columns as they are no longer needed
df.drop(['Date.of.Birth', 'DisbursalDate'], axis=1, inplace=True)

# Categorical encoding
le = LabelEncoder()
df['Employment.Type'] = le.fit_transform(df['Employment.Type'])

# Numerical scaling
scaler = StandardScaler()
df[['disbursed_amount', 'asset_cost']] = scaler.fit_transform(df[['disbursed_amount', 'asset_cost']])

# Drop unnecessary columns
df.drop(['UniqueID', 'branch_id', 'supplier_id'], axis=1, inplace=True)

# Define features and target variable
X = df.drop('loan_default', axis=1)
y = df['loan_default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = rf_clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
