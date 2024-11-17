# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from IPython.display import display

# Replace 'your_file_path' with the actual path to your file in Google Drive or local
file_path = 'final_hemoglobinopathy_analysis_dataset.csv'

# Read the data into a pandas DataFrame
data = pd.read_csv(file_path)

# Display first few rows with a clean table format
print(data.head())

# Defining the features and target
features = ['HbF', 'HbA', 'HbA2', 'HbS', 'Hb', 'PCV', 'RBC', 'MCV', 'MCH', 'RDW']
target = 'Hemoglobinopathy_Type'  # This column should contain the labels ('AA', 'AS', 'SS', 'Beta-Thalassemia Trait with AS')

# Convert target labels into numeric encoding (0 for AA, 1 for AS, 2 for SS, 3 for Beta-Thalassemia Trait with AS)
data[target] = data[target].map({'AA': 0, 'AS': 1, 'SS': 2, 'Beta-Thalassemia Trait with AS': 3})

# Split dataset into features (X) and target (y)
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
filename = 'random_forest_model(1).pkl'  # Use .pkl extension
joblib.dump(model, filename)
print(f"Model saved to {filename}")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
unique_classes = sorted(y_test.unique())
target_names = ['AA', 'AS', 'SS', 'Beta-Thalassemia Trait with AS']

# Adjust target names to match the unique classes
adjusted_target_names = [target_names[i] for i in unique_classes]

print(classification_report(y_test, y_pred, target_names=adjusted_target_names, zero_division=0))

# Visualize prediction probabilities for a new test case (custom values can be added)
test_case = pd.DataFrame([{
    'HbF': 0.5, 'HbA': 96, 'HbA2': 3.0, 'HbS': 10, 'Hb': 13, 'PCV': 40,
    'RBC': 5, 'MCV': 85, 'MCH': 30, 'RDW': 13
}])

# Predict the probabilities for the test case
probabilities = model.predict_proba(test_case)[0]

# Get class labels based on the trained model's classes
class_labels = model.classes_

# Create a bar chart for the prediction probabilities
plt.figure(figsize=(8, 6))
sns.barplot(x=adjusted_target_names, y=probabilities, palette="Blues_d", legend=False)
plt.title('Prediction Probabilities for Hemoglobinopathy')
plt.xlabel('Hemoglobinopathy Type')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.show()

# Show the predicted label for the test case
predicted_label = model.predict(test_case)[0]
predicted_class = adjusted_target_names[predicted_label]
print(f'The predicted hemoglobinopathy type for the test case is: {predicted_class}')

# Visualizing the chances of passing on each hemoglobinopathy type (hereditary chance)
hereditary_prob = {label: prob for label, prob in zip(adjusted_target_names, probabilities)}

# Plot hereditary chances
plt.figure(figsize=(8, 6))
plt.pie(hereditary_prob.values(), labels=hereditary_prob.keys(), autopct='%1.1f%%', colors=sns.color_palette("Blues", 4))
plt.title('Hereditary Chances of Passing Hemoglobinopathy')
plt.show()
