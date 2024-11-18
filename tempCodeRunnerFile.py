import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

try:
    model = load_model(r'sickle_cell_model_inveptionV3.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None