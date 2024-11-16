## README: **Integrating Deep Learning and HPLC Data to Detect and Predict Hereditary Transfer of Sickle Cell Anemia**

### **Overview**
This project aims to detect and predict Sickle Cell Anemia (SCA) using a combination of deep learning and High-Performance Liquid Chromatography (HPLC) data. It leverages cutting-edge deep learning models to classify SCA images and integrates Random Forest algorithms to analyze HPLC data for Hemoglobin F (HbF) prediction. By combining these approaches, the project provides an innovative method for accurate diagnosis and understanding of hereditary patterns in Sickle Cell Anemia.

---

### **Motivation**
Sickle Cell Anemia is a genetic blood disorder that affects the shape of red blood cells, causing them to become crescent-shaped, which reduces their ability to carry oxygen effectively. Early detection and prediction of disease severity are crucial for managing symptoms, improving quality of life, and providing tailored treatment options.

By integrating artificial intelligence techniques, this project aims to:
- Enhance diagnostic accuracy.
- Provide insights into the hereditary patterns of the disease.
- Facilitate early intervention and personalized healthcare strategies.

---

### **Objectives**
1. **Detection of Sickle Cell Anemia**:
   - Classify blood smear images as *Positive* (presence of disease) or *Negative* (absence of disease) using deep learning models.
   
2. **Prediction of Disease Severity**:
   - Analyze HPLC data to predict the severity of SCA based on Hemoglobin F (HbF) levels.

3. **Hereditary Analysis**:
   - Provide insights into the hereditary transfer patterns of SCA based on parental genetic traits.

---

### **Technologies Used**

1. **Deep Learning Models**:
   - **ResNet-50**, **Inception V3**, and **MobileNet** architectures were trained for image classification tasks.
   - The models were fine-tuned to achieve an impressive classification accuracy of *98.7%*.

2. **Random Forest Algorithm**:
   - Used for analyzing HPLC data to predict HbF levels and associated disease severity.

3. **Frameworks and Tools**:
   - **TensorFlow/Keras** for building and training deep learning models.
   - **Streamlit** for building an interactive web-based interface.
   - **Matplotlib** and **Seaborn** for data visualization.

4. **Data Sources**:
   - Blood smear images for SCA classification.
   - HPLC datasets containing Hemoglobin F (HbF) levels.

---

### **Features**

1. **Blood Smear Classification**:
   - Upload blood smear images to detect SCA using a fine-tuned **Inception V3** model.
   - Visualize the image and receive detailed probability scores for the classification results.

2. **HbF Level Prediction**:
   - Predict Hemoglobin F levels using HPLC data.
   - Classify patients into categories (*Normal*, *Mild*, or *Severe*) based on HbF levels.

3. **Hereditary Analysis**:
   - Visual representation of how genetic traits (e.g., SS, AS, and AA) are inherited from parents.
   - Help individuals understand their likelihood of passing the trait or disease to their offspring.

4. **User-Friendly Interface**:
   - A sleek, interactive **Streamlit**-based application that guides users through the entire detection and prediction process.
   - Visual feedback on uploaded images and results.

---

### **System Requirements**
1. **Hardware**:
   - GPU-enabled system (recommended) for faster inference.
   - Minimum 8GB RAM for smooth operation.

2. **Software**:
   - Python 3.8 or higher.
   - Required Python libraries:
     ```
     streamlit
     tensorflow
     numpy
     matplotlib
     seaborn
     pillow
     scikit-learn
     pandas
     ```

---

### **How to Run the Project**

#### **Step 1**: Clone the Repository
```bash
git clone https://github.com/<username>/sickle-cell-anemia-detection.git
cd sickle-cell-anemia-detection
```

#### **Step 2**: Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

#### **Step 3**: Prepare the Model Files
- Download the pre-trained deep learning models and place them in the appropriate directory (e.g., `templates/`).
- Ensure the file paths in the script match the locations of your model files.

#### **Step 4**: Run the Streamlit App
To start the Streamlit web application, execute:
```bash
streamlit run app.py
```
Replace `app.py` with the filename of your main script if different.

#### **Step 5**: Use the Application
- Upload a blood smear image for SCA classification.
- If the result is positive, proceed to the severity prediction module to predict HbF levels and understand associated health risks.

---

### **File Structure**
```
sickle-cell-anemia-detection/
│
├── templates/                # Contains pre-trained model files
│   ├── sickle_cell_model_inceptionV3.h5
│   ├── ... (other model files)
│
├── data/                     # Dataset files (HPLC and sample images)
│   ├── hplc_data.csv
│   ├── sample_images/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Required Python libraries
├── README.md                 # Documentation file (this file)
├── utils/                    # Helper functions for preprocessing and analysis
│
└── results/                  # Outputs and saved predictions
```

---

### **Results and Insights**
1. **Classification Accuracy**:
   - The deep learning models achieved an accuracy of **98.7%**, demonstrating robustness in SCA detection.

2. **Severity Prediction**:
   - The Random Forest model showed high reliability in predicting HbF levels and categorizing disease severity.

3. **Hereditary Analysis**:
   - Accurate prediction of genetic transfer patterns, helping users understand their likelihood of passing SCA traits.

---

### **Future Work**
1. **Integration of Additional Data**:
   - Include more patient data to improve model generalization and accuracy.

2. **Mobile App Development**:
   - Build a mobile application for real-time SCA detection.

3. **Explainable AI (XAI)**:
   - Implement XAI techniques to make the model’s predictions interpretable for medical professionals.

4. **Real-Time HPLC Integration**:
   - Connect with laboratory systems for automated HbF data analysis.

---

### **Acknowledgments**
- **Research Community**: For providing datasets and inspiration.
- **OpenAI GPT**: Assistance with documentation and technical support.
- **Collaborators and Mentors**: Their valuable insights and guidance throughout the project.

---

### **Contact**
For queries, please reach out to:
- **Name**: Mohammad Sufiyan
- **Email**: [your-email@example.com]
- **LinkedIn**: [Your LinkedIn Profile](#)

---

This project is an ongoing effort to use AI for improving healthcare. Contributions, suggestions, and collaborations are welcome!
