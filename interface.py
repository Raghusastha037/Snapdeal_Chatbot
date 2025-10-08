# interface.py
import streamlit as st
from c import SnapdealRAGChatbot
import os

st.set_page_config(page_title="Snapdeal AI Assistant", page_icon="🛍️")

st.title("🤖 Snapdeal AI Shopping Assistant")

# Get API key
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if not pinecone_api_key:
    pinecone_api_key = st.text_input("Enter your Pinecone API Key", type="password")

if pinecone_api_key:
    if "chatbot" not in st.session_state:
        with st.spinner("Initializing Assistant... (This may take ~1 min)"):
            st.session_state.chatbot = SnapdealRAGChatbot(pinecone_api_key)

    chatbot = st.session_state.chatbot

    st.markdown("💡 Try queries like: *smartphones under 15000*, *delivery policy*, *best laptops*")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat
    for role, msg in st.session_state.messages:
        st.chat_message(role).markdown(msg)

    # Chat input
    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        with st.spinner("Thinking..."):
            response = chatbot.chat(user_input)

        st.session_state.messages.append(("assistant", response))
        st.chat_message("assistant").markdown(response)
else:
    st.warning("⚠️ Please enter your Pinecone API key to continue.")
