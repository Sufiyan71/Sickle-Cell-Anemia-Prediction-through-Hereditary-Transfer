import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu

# Set the page title and layout
st.set_page_config(page_title="Sickle Cell Analysis", layout="wide")

# Sidebar title and info
st.sidebar.title("Sickle Cell Analysis")
st.sidebar.info(
    "Navigate through the application to detect sickle cell disease or check its severity. Use the menu below."
)

# Create a sidebar navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # Title of the navigation bar
        options=["Home", "Sickle Cell Detection", "Severity Check"],  # Options
        icons=["house", "activity", "bar-chart"],  # Corresponding icons
        menu_icon="cast",  # Icon for the menu title
        default_index=0,  # Default selected option
        styles = {
    "container": {"padding": "5px", "background-color": "#1F1F1F"},  # Dark background for the menu
    "icon": {"color": "white", "font-size": "25px"},  # White icons for better contrast
    "nav-link": {
        "font-size": "16px",
        "text-align": "left",
        "margin": "0px",
        "--hover-color": "#444",  # Darker shade on hover
        "border-radius": "8px",  # Rounded edges for links
        "box-shadow": "2px 2px 5px rgba(0, 0, 0, 0.5)",  # Shadow effect
    },
    "nav-link-selected": {
        # Dark shade for the selected button
        "color": "white",
        "box-shadow": "4px 4px 8px rgba(0, 0, 0, 0.8)",  # Intense shadow for active button
        "border-radius": "8px",  # Keep rounded edges
    },
}

    )

# Navigation logic
if selected == "Home":
    st.title("Title")
    st.write("# Integrating Deep Learning and Chromatographic Data to detect and predict hereditary Transfer of Sickle Cell Anemia")
    st.title("Problem Statement")
    st.write("### The innovation addresses the limited availability of accurate, accessible diagnostic tools for sickle cell anemia. Traditional methods like electrophoresis and blood smear analysis require expertise and resources that may be unavailable in low-resource regions. By incorporating deep learning models such as ResNet50, Mobile-Net, and InceptionV3, this solution enables efficient, high-accuracy detection from blood samples, potentially reducing the healthcare burden and enhancing patient care outcomes.")
    try:
        exec(open("app3.py").read())
    except FileNotFoundError:
        st.error("Error: app3.py not found.")
elif selected == "Sickle Cell Detection":
    # Redirect to app1.py
    try:
        exec(open("app1.py").read())
    except FileNotFoundError:
        st.error("Error: app1.py not found.")

elif selected == "Severity Check":
    # Redirect to app2.py
    try:
        exec(open("app2.py").read())
    except FileNotFoundError:
        st.error("Error: app2.py not found.")
