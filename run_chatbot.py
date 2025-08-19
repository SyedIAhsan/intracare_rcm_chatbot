#!/usr/bin/env python3
"""
One-command startup script for RCM Chatbot
This will process documents if needed and launch the Streamlit interface
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ['PINECONE_API_KEY', 'PINECONE_ENVIRONMENT', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please create a .env file with:")
        for var in missing_vars:
            print(f"  {var}=your_{var.lower()}")
        return False
    
    print("✅ Environment variables found")
    return True

def process_documents_if_needed():
    """Process documents if they haven't been processed yet"""
    print("📄 Checking if documents need processing...")
    
    # Add src to path so we can import our modules
    sys.path.append(str(Path(__file__).parent / "src"))
    
    try:
        from chatbot.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(pinecone_index_name="test-chatbot-docs")
        
        # Check if we have any vectors in the database
        try:
            stats = processor.get_database_stats()
            vector_count = stats.get('total_vector_count', 0)
            
            if vector_count > 0:
                print(f"✅ Found {vector_count} documents already processed")
                return True
            else:
                print("📚 No documents found in database, processing sample documents...")
                
        except Exception:
            print("📚 Database empty or inaccessible, processing sample documents...")
        
        # Process the data directory if it exists
        data_dir = Path("data")
        if data_dir.exists() and any(data_dir.iterdir()):
            print(f"🔄 Processing documents from {data_dir}...")
            result = processor.process_directory(str(data_dir))
            
            if result['success']:
                print(f"✅ Processed {result['total_chunks_parsed']} chunks")
                print(f"✅ Stored {result['vectors_upserted']} vectors")
                return True
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        # If no data directory, create and process test documents
        print("🔄 Creating and processing test documents...")
        from test_document_processing import create_test_documents
        
        test_dir = create_test_documents()
        result = processor.process_directory(test_dir)
        
        if result['success']:
            print(f"✅ Processed {result['total_chunks_parsed']} test document chunks")
            return True
        else:
            print(f"❌ Test document processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during document processing: {e}")
        return False

def launch_streamlit():
    """Launch the Streamlit application"""
    print("🚀 Starting Streamlit chatbot...")
    print("📱 Opening in your browser...")
    print("💬 Chat interface will be available shortly!")
    print("\n" + "="*50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.headless=false",
            "--server.port=8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped by user")
        return True
    
    return True

def main():
    """Main startup function"""
    print("🏥 RCM Chatbot Startup")
    print("=" * 30)
    
    # Step 1: Check environment
    if not check_environment():
        sys.exit(1)
    
    # Step 2: Process documents if needed
    if not process_documents_if_needed():
        print("❌ Document processing failed. Please check your configuration.")
        sys.exit(1)
    
    # Step 3: Launch Streamlit
    print("\n🎉 Everything ready! Launching chatbot...")
    launch_streamlit()

if __name__ == "__main__":
    main()