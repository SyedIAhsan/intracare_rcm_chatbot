#!/usr/bin/env python3
"""
Test script for document processing pipeline
Run this to test the complete flow from document parsing to Pinecone storage
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from chatbot.document_processor import DocumentProcessor

def create_test_documents():
    """Create sample documents for testing"""
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    # Create a test text file
    with open(test_dir / "sample.txt", "w") as f:
        f.write("""
        Revenue Cycle Management Best Practices
        
        Revenue cycle management (RCM) is the financial process that healthcare facilities 
        use to track patient care episodes from registration and appointment scheduling 
        to the final payment of a balance.
        
        Key components include:
        1. Patient registration and verification
        2. Insurance eligibility verification
        3. Charge capture and coding
        4. Claims submission
        5. Payment posting
        6. Denial management
        7. Patient collections
        
        Effective RCM requires proper staff training, technology integration, 
        and continuous monitoring of key performance indicators.
        """)
    
    # Create another test file
    with open(test_dir / "healthcare_info.txt", "w") as f:
        f.write("""
        Healthcare Documentation Standards
        
        Proper documentation is essential for healthcare providers to ensure 
        accurate billing and compliance with regulations.
        
        Documentation must include:
        - Patient demographics
        - Insurance information
        - Diagnosis codes (ICD-10)
        - Procedure codes (CPT)
        - Medical necessity justification
        
        Quality documentation supports proper reimbursement and reduces 
        the risk of audits and denials.
        """)
    
    print(f"✅ Created test documents in {test_dir}")
    return str(test_dir)

def test_complete_pipeline():
    """Test the complete document processing pipeline"""
    
    print("🚀 Starting Document Processing Pipeline Test\n")
    
    # Step 1: Check environment variables
    print("1️⃣  Checking environment variables...")
    required_vars = ['PINECONE_API_KEY', 'PINECONE_ENVIRONMENT', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set these in your environment or .env file")
        return False
    print("✅ All environment variables found\n")
    
    # Step 2: Create test documents
    print("2️⃣  Creating test documents...")
    test_dir = create_test_documents()
    print()
    
    # Step 3: Initialize processor
    print("3️⃣  Initializing document processor...")
    try:
        processor = DocumentProcessor(pinecone_index_name="test-chatbot-docs")
        print("✅ Document processor initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return False
    
    # Step 4: Process documents
    print("4️⃣  Processing test documents...")
    try:
        result = processor.process_directory(test_dir)
        if result['success']:
            print(f"✅ Processed {result['total_chunks_parsed']} chunks")
            print(f"✅ Upserted {result['vectors_upserted']} vectors to Pinecone")
            print(f"⚠️  Failed: {result['vectors_failed']} vectors\n")
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return False
    
    # Step 5: Test search functionality
    print("5️⃣  Testing search functionality...")
    test_queries = [
        "revenue cycle management",
        "documentation standards",
        "ICD-10 codes"
    ]
    
    for query in test_queries:
        try:
            results = processor.search_documents(query, top_k=3)
            print(f"Query: '{query}' -> Found {len(results)} results")
            if results:
                print(f"  Best match score: {results[0]['score']:.4f}")
                print(f"  Content preview: {results[0]['content'][:100]}...")
            print()
        except Exception as e:
            print(f"❌ Search error for '{query}': {e}")
            return False
    
    # Step 6: Show database stats
    print("6️⃣  Database statistics...")
    try:
        stats = processor.get_database_stats()
        print(f"✅ Total vectors in database: {stats.get('total_vector_count', 'Unknown')}")
        print(f"✅ Database dimension: {stats.get('dimension', 'Unknown')}")
        print()
    except Exception as e:
        print(f"⚠️  Could not get stats: {e}\n")
    
    print("🎉 Pipeline test completed successfully!")
    print("\nWhat you should see:")
    print("- Documents parsed into text chunks")
    print("- Each chunk converted to embeddings")
    print("- Vectors stored in Pinecone")
    print("- Search returning relevant results with similarity scores")
    
    return True

if __name__ == "__main__":
    success = test_complete_pipeline()
    if not success:
        sys.exit(1)