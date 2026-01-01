import streamlit as st
import requests

# --- CONFIGURATION ---
API_BASE_URL = "http://localhost:8000" # Change to Render URL after deployment

st.set_page_config(page_title="SkinCare AI", page_icon="✨", layout="centered")

# --- CUSTOM EYE-CATCHING UI (CSS) ---
st.markdown("""
    <style>
    /* 1. Background Color */
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #FDF6F0 100%);
    }
    
    /* 2. Style the Title */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        color: #4A4A4A;
        font-weight: 700;
        font-size: 42px;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    
    /* 3. Subtitle */
    .sub-title {
        text-align: center;
        color: #8E8E8E;
        font-size: 16px;
        margin-bottom: 30px;
    }

    /* 4. Chat Bubble Styling */
    [data-testid="stChatMessage"] {
        background-color: white !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #F0F0F0;
        margin-bottom: 15px;
    }

    /* 5. Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 6. Button Style */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #D1D1D1;
        background-color: white;
        color: #4A4A4A;
        transition: 0.3s;
    }
    .stButton>button:hover {
        border-color: #A0C4FF;
        color: #A0C4FF;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<h1 class="main-title">✨ SkinCare AI</h1>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📍 Source: {', '.join(msg['sources'])}")

# --- CHAT INPUT ---
if prompt := st.chat_input("How can I help you today?"):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                r = requests.post(f"{API_BASE_URL}/chat", json={"query": prompt})
                if r.status_code == 200:
                    data = r.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        st.caption(f"📍 Source: {', '.join(sources)}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                else:
                    st.error("Server is currently busy. Try again soon.")
            except Exception as e:
                st.error("Connection Error. Please check if backend is running.")

# --- FOOTER BUTTONS ---
cols = st.columns([4, 1])
with cols[1]:
    if st.button("🗑️ Clear"):
        st.session_state.messages = []
        st.rerun()