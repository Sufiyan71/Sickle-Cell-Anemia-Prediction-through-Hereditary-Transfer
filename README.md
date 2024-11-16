Here's an updated README that reflects the use of **Streamlit** instead of React for the web application:

---

# Sickle Cell Detection Using InceptionV3

This project implements a deep learning pipeline for detecting sickle cell anemia (SCA) using blood smear images. By leveraging the **InceptionV3** architecture, a pre-trained Convolutional Neural Network (CNN), we classify images as either positive (sickle cells detected) or negative (normal cells). A **Streamlit-based web application** is included for an intuitive user interface.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Setup Instructions](#setup-instructions)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## **Project Overview**
Sickle cell anemia is a genetic disorder affecting the shape and functionality of red blood cells. Crescent or sickle-shaped cells can block blood flow, leading to severe complications. This project aims to:
1. Provide a **deep learning-based solution** for early detection using blood smear images.
2. Build a **Streamlit-based web application** for seamless interaction.
3. Achieve high accuracy and reliability in detecting SCA to support medical professionals.

---

## **Key Features**
- **Image Classification**: Upload blood smear images to classify as positive or negative for sickle cell anemia.
- **Interactive Web Application**:
  - Developed in **Streamlit** for an easy-to-use interface.
  - Allows for real-time predictions and visualizations.
- **Visualization**: Probability scores and predictions are visualized with bar charts.
- **High Accuracy**: The model achieves a classification accuracy of **98.89%**, with strong precision, recall, and F1 scores.

---

## **Technologies Used**
- **Python** (for model development and application backend)
- **TensorFlow/Keras** (model building and training)
- **Streamlit** (interactive web application)
- **Matplotlib** and **Seaborn** (visualizations)
- **ImageDataGenerator** (data augmentation for training)

---

## **Dataset**
The dataset consists of labeled blood smear images divided into two categories:
- **Positive**: Images with visible sickle cells.
- **Negative**: Images of normal blood cells.

### **Dataset Preparation**
- Ensure the dataset is organized into subfolders:
  ```
  /data
    /Positive
    /Negative
  ```
- Images will be automatically split into training, validation, and testing sets during preprocessing.

### **Data Augmentation**
To enhance the robustness of the model, we used **ImageDataGenerator** for:
- Horizontal and vertical flips.
- Random rotations.
- Scaling and zooming.

---

## **Model Training**
The **InceptionV3** model was fine-tuned with custom layers for binary classification. Key steps include:
1. **Transfer Learning**:
   - Pre-trained on ImageNet for feature extraction.
   - Fine-tuned on the blood smear dataset for SCA detection.
2. **Loss Function**:
   - Binary cross-entropy.
3. **Optimizer**:
   - Adam optimizer with a learning rate of 0.001.

### **Training Pipeline**
1. Load and preprocess the dataset.
2. Perform data augmentation.
3. Fine-tune the InceptionV3 model.
4. Evaluate performance metrics.

---

## **Web Application**
The web application is built with **Streamlit**, enabling users to interact with the model in real time.

### **Features**
- **Image Upload**: Upload a medical image of blood cells.
- **Model Prediction**: The application displays the classification (positive/negative) with probabilities.
- **Probability Visualization**: A bar chart visualizes the confidence scores for each class.

---

## **Setup Instructions**

### **Step 1**: Clone the Repository
```bash
git clone https://github.com/<username>/sickle-cell-detection.git
cd sickle-cell-detection
```

### **Step 2**: Install Dependencies
Create a virtual environment and install required libraries:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### **Step 3**: Prepare the Dataset
Organize your dataset as described in the [Dataset](#dataset) section.

### **Step 4**: Train the Model
If you want to retrain the model, use the training script:
```bash
python train_model.py
```

### **Step 5**: Run the Streamlit App
Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

---

## **Results**
The trained **InceptionV3** model achieved the following metrics:
- **Accuracy**: 98.89%
- **Precision**: 1.0
- **Recall**: 1.0
- **F1 Score**: 1.0

---

## **Contributing**
We welcome contributions to enhance the project. Follow these steps to contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Create a pull request.

---

## **License**
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the project for personal or commercial use with attribution.

---

## **Contact**
For inquiries, reach out to:
- **Name**: Mohammad Sufiyan
- **Email**: [mohammadsufiyansheikh@gmail.com]
- **LinkedIn**: [https://www.linkedin.com/in/sufiyan-sheikh-0088a724a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app](#)

---

Feel free to explore and improve this project. Together, we can make significant progress in healthcare through AI!
