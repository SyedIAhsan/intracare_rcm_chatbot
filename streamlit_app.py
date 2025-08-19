import streamlit as st
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from chatbot.rag_chatbot import RAGChatbot

st.set_page_config(
    page_title="RCM Chatbot",
    page_icon="🏥",
    layout="centered"
)

# Initialize chatbot
@st.cache_resource
def initialize_chatbot():
    return RAGChatbot(index_name="test-chatbot-docs")

def main():
    st.title("🏥 RCM Chatbot")
    st.caption("Ask questions about Revenue Cycle Management")

    # Initialize chatbot
    try:
        chatbot = initialize_chatbot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.info("Make sure your .env file is configured with API keys")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"• **{source['source']}** (Score: {source['score']:.3f})")

    # Chat input
    if prompt := st.chat_input("Ask about RCM processes..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    result = chatbot.chat(prompt)
                    response = result["response"]
                    sources = result["sources"]
                    
                    st.markdown(response)
                    
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.write(f"• **{source['source']}** (Score: {source['score']:.3f})")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This chatbot uses your uploaded documents to answer questions about Revenue Cycle Management.")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()