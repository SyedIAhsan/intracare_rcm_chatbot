import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

try:
    from .document_processor import DocumentProcessor
except ImportError:
    from document_processor import DocumentProcessor

load_dotenv()
logger = logging.getLogger(__name__)


class RAGChatbot:
    def __init__(
        self, index_name: str = "chatbot-docs", max_context_chunks: int = 5
    ):
        self.processor = DocumentProcessor(pinecone_index_name=index_name)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_context_chunks = max_context_chunks

        self.system_prompt = """You are an expert assistant for Revenue Cycle Management (RCM) and Intracare policies. 

Use the provided context documents to answer questions accurately. If the context doesn't contain enough information to answer the question, say so clearly.

Always cite which document or section your answer comes from when possible.

Be helpful, professional, and concise in your responses."""

    def search_relevant_docs(
        self, query: str, top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents based on the query"""
        if top_k is None:
            top_k = self.max_context_chunks

        results = self.processor.search_documents(query, top_k=top_k)
        return results

    def build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Build context string from search results"""
        if not search_results:
            return "No relevant documents found."

        context_parts = []
        for i, result in enumerate(search_results, 1):
            source = result["metadata"].get("source", "Unknown source")
            content = result["content"]
            score = result["score"]

            context_parts.append(
                f"Document {i} (Score: {score:.3f}, Source: {source}):\n{content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI with context"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Context documents:
{context}

---

Question: {query}

Please answer based on the context provided above.""",
            },
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error generating a response: {str(e)}"

    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat function - search docs and generate response"""
        logger.info(f"Processing query: {query}")

        # Search for relevant documents
        search_results = self.search_relevant_docs(query)

        if not search_results:
            return {
                "query": query,
                "response": "I couldn't find any relevant documents to answer your question. Please try rephrasing or asking about a different topic.",
                "sources": [],
                "context_used": "",
            }

        # Build context from search results
        context = self.build_context(search_results)

        # Generate response
        response = self.generate_response(query, context)

        # Extract sources
        sources = []
        for result in search_results:
            sources.append(
                {
                    "source": result["metadata"].get("source", "Unknown"),
                    "score": result["score"],
                    "chunk_id": result["metadata"].get("chunk_id", 0),
                }
            )

        return {
            "query": query,
            "response": response,
            "sources": sources,
            "context_used": context,
        }


def main():
    """Command line interface for the RAG chatbot"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Chatbot for RCM documents")
    parser.add_argument("--query", help="Ask a question")
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive chat"
    )
    parser.add_argument(
        "--index", default="chatbot-docs", help="Pinecone index name"
    )

    args = parser.parse_args()

    chatbot = RAGChatbot(index_name=args.index)

    if args.query:
        # Single query mode
        result = chatbot.chat(args.query)
        print(f"Question: {result['query']}")
        print(f"Answer: {result['response']}")
        print(f"\nSources used:")
        for source in result["sources"]:
            print(f"- {source['source']} (Score: {source['score']:.3f})")

    elif args.interactive:
        # Interactive mode
        print("RAG Chatbot - Ask questions about your documents!")
        print("Type 'quit' to exit\n")

        while True:
            query = input("You: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            result = chatbot.chat(query)
            print(f"\nBot: {result['response']}")

            if result["sources"]:
                print(f"\nSources:")
                for source in result["sources"][:3]:  # Show top 3 sources
                    print(f"  • {source['source']} (relevance: {source['score']:.2f})")
            print()

    else:
        print("Use --query 'your question' or --interactive")


if __name__ == "__main__":
    main()
