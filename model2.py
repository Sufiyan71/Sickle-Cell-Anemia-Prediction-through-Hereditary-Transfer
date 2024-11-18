import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate
import joblib
from IPython.display import display

# Replace 'your_file_path' with the actual path to your file
file_path = 'final_hemoglobinopathy_analysis_dataset.csv'

# Read the data into a pandas DataFrame
data = pd.read_csv(file_path)

# Display first few rows with a clean table format
display(data.head())

# Feature columns (assuming these are the columns in your dataset)
features = ['HbS', 'HbA', 'HbA2', 'PCV', 'MCV', 'MCH']

# Function to derive severity and health problems
def create_target_and_problems(row):
    problems = []

    if row['HbS'] > 13:
        problems.append('Sickle cell disease or trait')

    if row['HbA'] > 98:
        problems.append('Polycythemia (high red blood cells)')

    if row['HbA2'] > 3.3:
        problems.append('Beta-thalassemia trait')

    if row['PCV'] > 60:
        problems.append('Polycythemia vera or dehydration')

    if row['MCV'] > 96:
        problems.append('Macrocytic anemia (Vitamin B12 or folate deficiency)')

    if row['MCH'] > 34:
        problems.append('Macrocytic anemia or hyperchromic anemia')

    # Set severity based on number of problems
    severity = 'Severe' if len(problems) > 2 else 'Mild' if len(problems) > 0 else 'Healthy'
    return pd.Series([severity, problems])

# Apply function to create the target and problems columns
data[['Severity', 'Health_Problems']] = data.apply(create_target_and_problems, axis=1)

# Encode target (if necessary)
data['Severity'] = data['Severity'].map({'Healthy': 0, 'Mild': 1, 'Severe': 2})

# Split dataset into training and testing sets
X = data[features]
y = data['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
filename = 'random_forest_model(2).pkl'
joblib.dump(model, filename)
print(f"Model saved to {filename}")

# Evaluate Model Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Mild', 'Severe'], zero_division=0))

# Real-Time Analysis with a custom test case (replace with actual patient data)
test_case = pd.DataFrame([{
    'HbS': 15,         # Example value
    'HbA': 100,        # Example value
    'HbA2': 3.5,       # Example value
    'PCV': 65,         # Example value
    'MCV': 98,         # Example value
    'MCH': 36          # Example value
}])

# Predict the severity for the test case
real_time_prediction = model.predict(test_case)

# Map severity
severity_mapping = {0: 'Healthy', 1: 'Mild', 2: 'Severe'}
severity_result = severity_mapping[real_time_prediction[0]]

# Calculate health problems for the test case using the same logic
test_case[['Severity', 'Health_Problems']] = create_target_and_problems(test_case.iloc[0]).to_frame().T  # Convert the result to a DataFrame with the correct shape

# Convert the DataFrame to a dictionary for tabulate output
test_case_tabulate = test_case.to_dict(orient='records')

# Display output using tabulate
print("\n--- Patient Test Case Result ---\n")
print(tabulate(test_case_tabulate, headers='keys', tablefmt='grid'))

# Output the severity level and health problems
print(f'\nThe predicted severity is: {severity_result}')
print(f'Potential health problems: {test_case["Health_Problems"].values[0]}')
