import streamlit as st
from pages import about_me
from pages import login_page

# st.markdown("""
#     <style>
#     [data-testid="stSidebarNav"] {display: none;}
#     </style>
# """, unsafe_allow_html=True)

st.set_page_config(page_title="Visual-AI", page_icon="🤖", layout="wide")

if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = False
if "language" not in st.session_state:
    st.session_state.language = "Română"

def apply_styles():
    bg_color = "#000000" if st.session_state.high_contrast else "#121212"
    text_color = "#FFD700" if st.session_state.high_contrast else "#FFFFFF"
    button_color = "#FFA500" if st.session_state.high_contrast else "#4CAF50"

    st.markdown(f"""
        <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .contrast-switch {{
            display: inline-flex;
            align-items: center;
            cursor: pointer;
            padding: 6px 10px;
            background-color: {button_color};
            color: black;
            font-size: 12px;
            border-radius: 20px;
            border: none;
            transition: background-color 0.3s;
        }}
        .contrast-switch:hover {{
            background-color: #FF8C00;
        }}
        .stSidebar {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: {text_color} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

apply_styles()

translations = {
    "Română": {
        "home_title": "🏠 Acasă",
        "about": "👤 Despre Mine",
        "login": "🔑 Autentificare",
        "language": "🌐 Schimbă limba",
        "welcome_message": "Bine ai venit în Visual-AI!",
        "app_description": """
        **Visual-AI** este o aplicație destinată persoanelor cu deficiențe de vedere, ajutându-le să navigheze în siguranță.  
        Aceasta folosește **inteligența artificială** pentru:  
        - Detectarea obiectelor  
        - Estimarea distanțelor  
        - Ghidare vocală  

        **Funcționalități viitoare:**  
        ✅ Aplicație **iOS** cu senzor **LiDAR** pentru precizie mai mare  
        ✅ Integrare cu **Apple Maps** pentru navigare ușoară către stații, magazine etc.  
        """,
        "contrast_button": "🎨 Activează/Dezactivează High-Contrast"
    },
    "English": {
        "home_title": "🏠 Home",
        "about": "👤 About Me",
        "login": "🔑 Login",
        "language": "🌐 Change Language",
        "welcome_message": "Welcome to Visual-AI!",
        "app_description": """
        **Visual-AI** is an application designed to assist visually impaired individuals in navigating safely.  
        It utilizes **artificial intelligence** to:  
        - Detect objects  
        - Estimate distances  
        - Provide voice guidance  

        **Future Features:**  
        ✅ **iOS app** with **LiDAR sensor** for higher accuracy  
        ✅ **Apple Maps integration** for easy navigation to stations, stores, etc.  
        """,
        "contrast_button": "🎨 Toggle High-Contrast"
    }
}

if "language" not in st.session_state:
    st.session_state.language = "Română"

language = st.sidebar.selectbox(
    "🌍 Selectează limba / Choose Language:", ["Română", "English"],
    index=0 if st.session_state.language == "Română" else 1
)
st.session_state.language = language
t = translations[language]

st.sidebar.title("🔎 Navigare")
page = st.sidebar.radio("", [t["home_title"], t["about"], t["login"]])

if st.sidebar.button(t["contrast_button"]):
    st.session_state.high_contrast = not st.session_state.high_contrast
    apply_styles()
    st.rerun()

if page == t["home_title"]:
    st.title(t["welcome_message"])
    st.markdown(t["app_description"])
elif page == t["about"]:
    about_me.show_about_page()
elif page == t["login"]:
    login_page.show_login_page()