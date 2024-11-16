import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-trained models
model_classification = joblib.load(r'random_forest_model(1).pkl')
model_severity = joblib.load(r'random_forest_model(2).pkl')

# App title
st.title("Hemoglobinopathy Classification and HbF Severity Prediction")

# User input fields
HbF = st.number_input("HbF (%)", min_value=0.0, format="%.2f")
HbA = st.number_input("HbA (%)", min_value=0.0, format="%.2f")
HbA2 = st.number_input("HbA2 (%)", min_value=0.0, format="%.2f")
HbS = st.number_input("HbS (%)", min_value=0.0, format="%.2f")
Hb = st.number_input("Hemoglobin (g/dL)", min_value=0.0, format="%.2f")
PCV = st.number_input("Packed Cell Volume (%)", min_value=0.0, format="%.2f")
RBC = st.number_input("Red Blood Cell Count (million/µL)", min_value=0.0, format="%.2f")
MCV = st.number_input("Mean Corpuscular Volume (fL)", min_value=0.0, format="%.2f")
MCH = st.number_input("Mean Corpuscular Hemoglobin (pg)", min_value=0.0, format="%.2f")
RDW = st.number_input("Red Cell Distribution Width (%)", min_value=0.0, format="%.2f")

# Custom label mapping
custom_labels = {0: 'AA', 1: 'AS', 2: 'SS', 3: 'Beta-Thalassemia Trait with AS'}
severity_classes = {
    0: 'Healthy',
    1: 'Mild or Healthy (Carrier, no major symptoms)',
    2: 'Severe (Significant symptoms and health issues)'
}

# Collect features for classification
features_classification = [[HbF, HbA, HbA2, HbS, Hb, PCV, RBC, MCV, MCH, RDW]]

if st.button("Classify Hemoglobinopathy and Predict Severity"):
    try:
        # Step 1: Hemoglobinopathy Classification
        hemoglobinopathy_prediction = model_classification.predict(features_classification)[0]
        hemoglobinopathy_classes = model_classification.classes_
        classification_result = custom_labels[hemoglobinopathy_prediction]
        st.write(f"**Predicted Hemoglobinopathy Type:** {classification_result}")

        # Display prediction probabilities
        probabilities = model_classification.predict_proba(features_classification)[0]
        fig, ax = plt.subplots()
        sns.barplot(x=hemoglobinopathy_classes, y=probabilities, palette="Blues_d", ax=ax)
        ax.set_title('Prediction Probabilities for Hemoglobinopathy')
        ax.set_xlabel('Hemoglobinopathy Type')
        ax.set_ylabel('Probability')
        st.pyplot(fig)

        # Step 2: Severity Prediction
        features_severity = [[HbS, HbA, HbA2, PCV, MCV, MCH]]
        severity_prediction = model_severity.predict(features_severity)[0]
        severity_result = severity_classes.get(severity_prediction, "Unknown")
        st.write(f"**Predicted Severity Level:** {severity_result}")

        # Step 3: Health Problems
        def create_health_problems(HbS, HbA, HbA2, PCV, MCV, MCH):
            problems = []
            if HbS > 13:
                problems.append('Sickle cell disease or trait')
            if HbA > 98:
                problems.append('Polycythemia (high red blood cells)')
            if HbA2 > 3.3:
                problems.append('Beta-thalassemia trait')
            if PCV > 60:
                problems.append('Polycythemia vera or dehydration')
            if MCV > 96:
                problems.append('Macrocytic anemia (Vitamin B12 or folate deficiency)')
            if MCH > 34:
                problems.append('Macrocytic anemia or hyperchromic anemia')
            return problems

        health_problems = create_health_problems(HbS, HbA, HbA2, PCV, MCV, MCH)
        st.write("**Potential Health Problems:**")
        if health_problems:
            for problem in health_problems:
                st.write(f"- {problem}")
        else:
            st.write("No health problems detected based on input values.")

        # Step 4: Hereditary Chances
        hereditary_prob = {label: prob for label, prob in zip(hemoglobinopathy_classes, probabilities)}
        fig2, ax2 = plt.subplots()
        ax2.pie(hereditary_prob.values(), labels=hereditary_prob.keys(), autopct='%1.1f%%', colors=sns.color_palette("Blues", len(hemoglobinopathy_classes)))
        ax2.set_title('Hereditary Chances of Passing Hemoglobinopathy')
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
