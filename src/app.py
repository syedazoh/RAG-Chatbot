import streamlit as st
import os
import sys

# Step 1: Tell Python where to find your 'core' folder
sys.path.append(os.path.join(os.getcwd(), "src"))

# Step 2: Import your logic directly
try:
    from core.rag_pipeline import get_rag_chain
except ImportError as e:
    st.error(f"Could not find core/rag_pipeline.py. Check your folder structure! Error: {e}")
    st.stop()

# --- UI CONFIG ---
st.set_page_config(page_title="SkinCare AI", page_icon="✨", layout="centered")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #FFFFFF 0%, #FDF6F0 100%); }
    .main-title { font-family: 'Helvetica Neue'; color: #4A4A4A; font-weight: 700; font-size: 42px; text-align: center; }
    [data-testid="stChatMessage"] { background-color: white !important; border-radius: 20px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">✨ SkinCare AI</h1>', unsafe_allow_html=True)

# --- ENGINE LOADING ---
if "rag_chain" not in st.session_state:
    with st.spinner("🔍 Loading skincare knowledge base..."):
        # This calls your rag_pipeline.py directly in this process
        st.session_state.rag_chain = get_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DISPLAY MESSAGES ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📍 Source: {', '.join(msg['sources'])}")

# --- CHAT INPUT ---
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("📚 Searching documents..."):
            try:
                # Direct function call to the RAG logic
                result = st.session_state.rag_chain(prompt)
                
                answer = result.get("answer", "I couldn't find an answer.")
                sources = result.get("sources", [])
                
                st.markdown(answer)
                if sources:
                    st.caption(f"📍 Source: {', '.join(sources)}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources
                })
            except Exception as e:
                st.error(f"RAG Engine Error: {e}")

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()