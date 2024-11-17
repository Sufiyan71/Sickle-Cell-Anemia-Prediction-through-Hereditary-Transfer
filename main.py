import streamlit as st

# Define the pages for your multipage app
PAGES = {
    "Sickle Cell Detection": "app1.py",
    "Severity Check": "app2.py"
}

# Apply custom CSS for sidebar styling
st.markdown(
    """
    <style>
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #2C3E50;
            color: white;
        }
        .css-10trblm {
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #2C3E50;
            padding: 10px;
        }
        .sidebar h2 {
            color: #FFFFFF;
        }
        .sidebar .sidebar-item {
            padding: 5px;
        }
        /* Active page highlight */
        .css-1dp5vir:hover {
            background-color: #34495E;
            border-radius: 5px;
            color: #FFFFFF;
        }
        .css-1dp5vir {
            color: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for navigation with title and description
st.sidebar.title("Sickle Cell Analysis")
st.sidebar.info(
    "Navigate through the application to detect sickle cell disease or check its severity. Use the menu below."
)

# Initialize session state keys if not already set
if "redirect_to_app11" not in st.session_state:
    st.session_state["redirect_to_app11"] = False

# Sidebar for selecting a page
selection = st.sidebar.radio("Choose a Page", list(PAGES.keys()))

# Function to load selected app
def load_app(page_name):
    try:
        exec(open(PAGES[page_name]).read())
    except FileNotFoundError:
        st.error(f"The page '{page_name}' could not be loaded.")

# Automatically redirect to 'Severity Check' if session state is set
if st.session_state["redirect_to_app11"]:
    st.session_state["redirect_to_app11"] = False  # Reset the flag after redirection
    load_app("Severity Check")
else:
    # Load the selected app
    load_app(selection)
