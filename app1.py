import numpy as np
import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image


model = load_model(r'sickle_cell_model_inveptionV3.h5')

# Custom function to load and predict the label for the image
def predict(img):
    # Step 1: Resize the image to 299x299 as required by InceptionV3
    img = img.resize((299, 299))
    
    # Step 2: Convert image to numpy array (shape must be (299, 299, 3))
    import numpy as np
    img_array = np.array(img)

    # Step 3: Ensure the array has 3 color channels
    if img_array.shape[-1] == 4:
       img_array = img_array[..., :3]  # Convert RGBA to RGB if needed

    # Step 4: Normalize the pixel values between 0 and 1
    img_array = img_array / 255.0

    # Step 5: Reshape the image array to add batch dimension (1, 299, 299, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Step 6: Plot the Loaded Image
    st.image(img, caption="Loaded Image", use_column_width=True)

    # Step 7: Make the prediction
    p = model.predict(img_array)

    # Step 8: Define the label array
    labels = {0: 'Negative', 1: 'Positive'}

    # Step 9: Display the result
    positive_prob = round(p[0][1] * 100, 2)  # Probability of the positive class (index 1)
    predicted_class = labels[np.argmax(p[0], axis=-1)]  # Get predicted class
    
    # Step 10: Display individual probabilities
    classes = []
    prob = []
    for i, j in enumerate(p[0]):
        classes.append(labels[i])
        prob.append(round(j * 100, 2))
      
    for i, j in enumerate(p[0]):
        st.write(f"{labels[i].upper()}: {round(j * 100, 2)} %")
        classes.append(labels[i])
        prob.append(round(j * 100, 2))
      
    # Step 11: Plotting the probabilities
    fig, ax = plt.subplots()
    ax.bar(classes, prob)
    ax.set_xlabel('Labels')
    ax.set_ylabel('Probability')
    ax.set_title('Probability for loaded image')
    st.pyplot(fig)

    # Display the positive probability with bold formatting
    st.write(f"There is <b><i>{positive_prob}%</i></b> chance that you may have the disease.", unsafe_allow_html=True)

    # Display the predicted class with bold formatting
    st.write(f"Your Report is <b><i>{predicted_class}</i></b>", unsafe_allow_html=True)

    if predicted_class == 'Positive':
        if st.button("Check Severity"):
            #st.experimental_set_query_params(app='app5') 
            # Redirect the user to app2 (app5)
            st.session_state['redirect_to_app2'] = True 
            st.write("Redirecting to next page...")# Set the app to app5 (app2)
            st.rerun() 
             
            

# Streamlit App Frontend
st.title("Sickle Cell Detection using InceptionV3")
# Custom styling for the drag-and-drop area
st.markdown("""
    <style>
        /* Custom bordered rectangle around the file uploader */
        .file-uploader-container {
            border: 4px dashed #ffffff;
            border-radius: 10px;
            padding: 20px;
            color: #3f51b5;
            font-size: 16px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .file-uploader-container:hover {
            background-color: #;
        }
        /* Footer styling */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #333;
            color: #dddddd;
            text-align: center;
            padding: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Styled file uploader container
st.markdown("""
<div class="file-uploader-container">
    <h4 style="color: #ffffff;">Click to upload or drag and drop an image</h4>
    <p style="font-size: 14px; color: #666;">Accepted formats: jpg, jpeg, png</p>
</div>
""", unsafe_allow_html=True)
# Styled file uploader container
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# File submission logic
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    if st.button("Submit", key="submit_button"):
        img = Image.open(uploaded_file)
        predict(img)  # Assuming the `predict` function is defined to handle the prediction
 
# Redirect to app5 if set in session state
if 'app' in st.session_state and st.session_state['app'] == 'app2':
    st.rerun()  # Rerun to load app5
    

