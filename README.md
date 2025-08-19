# Intracare RCM Chatbot

RAG-powered chatbot for Revenue Cycle Management using OpenAI and Pinecone.

## Quick Start

1. **Setup environment:**
   ```bash
   conda create -n chatbot python=3.11 -y
   conda activate chatbot
   pip install -r requirements.txt
   ```

2. **Create `.env` file:**
   ```
   OPENAI_API_KEY=your_key_here
   PINECONE_API_KEY=your_key_here
   PINECONE_ENVIRONMENT=your_env_here
   ```

3. **Run chatbot:**
   ```bash
   python run_chatbot.py
   ```

## Other Commands

```bash
# Feed data to system
python -m src.chatbot.document_processor --directory data

# Interactive terminal chat
python -m src.chatbot.rag_chatbot --interactive

# Single query
python -m src.chatbot.rag_chatbot --query "What is RCM?"

# Manual Streamlit launch
streamlit run streamlit_app.py
```

## How It Works

- Processes documents from `/data` folder
- Stores embeddings in Pinecone vector database
- Uses OpenAI GPT for responses with retrieved context
- Provides web interface via Streamlit