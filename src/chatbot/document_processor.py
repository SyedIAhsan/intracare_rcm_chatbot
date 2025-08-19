import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .document_parser import DocumentParser
from .pinecone_handler import PineconeHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, pinecone_index_name: str = "chatbot-docs", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.parser = DocumentParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.pinecone_handler = PineconeHandler(index_name=pinecone_index_name)
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        logger.info(f"Processing file: {file_path}")
        
        try:
            chunks = self.parser.parse_document(file_path)
            logger.info(f"Parsed {len(chunks)} chunks from {file_path}")
            
            result = self.pinecone_handler.upsert_documents(chunks)
            logger.info(f"Upserted {result['upserted']} vectors to Pinecone")
            
            return {
                'file_path': file_path,
                'chunks_parsed': len(chunks),
                'vectors_upserted': result['upserted'],
                'vectors_failed': result['failed'],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                'file_path': file_path,
                'error': str(e),
                'success': False
            }
    
    def process_directory(self, directory_path: str, file_extensions: List[str] = None) -> Dict[str, Any]:
        logger.info(f"Processing directory: {directory_path}")
        
        try:
            chunks = self.parser.parse_directory(directory_path, file_extensions)
            logger.info(f"Parsed {len(chunks)} total chunks from directory")
            
            result = self.pinecone_handler.upsert_documents(chunks)
            logger.info(f"Upserted {result['upserted']} vectors to Pinecone")
            
            return {
                'directory_path': directory_path,
                'total_chunks_parsed': len(chunks),
                'vectors_upserted': result['upserted'],
                'vectors_failed': result['failed'],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            return {
                'directory_path': directory_path,
                'error': str(e),
                'success': False
            }
    
    def search_documents(self, query: str, top_k: int = 5, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        filter_dict = {'source': source_filter} if source_filter else None
        return self.pinecone_handler.search(query, top_k=top_k, filter_dict=filter_dict)
    
    def delete_document(self, file_path: str) -> bool:
        return self.pinecone_handler.delete_by_source(file_path)
    
    def get_database_stats(self) -> Dict[str, Any]:
        return self.pinecone_handler.get_index_stats()

def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents for chatbot")
    parser.add_argument("--file", help="Process a single file")
    parser.add_argument("--directory", help="Process all files in a directory")
    parser.add_argument("--search", help="Search the document database")
    parser.add_argument("--delete", help="Delete a document by file path")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    
    args = parser.parse_args()
    
    processor = DocumentProcessor(pinecone_index_name="chatbot-docs")
    
    if args.file:
        result = processor.process_single_file(args.file)
        print(f"Result: {result}")
    
    elif args.directory:
        result = processor.process_directory(args.directory)
        print(f"Result: {result}")
    
    elif args.search:
        results = processor.search_documents(args.search)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   Source: {result['metadata']['source']}")
            print(f"   Content: {result['content'][:200]}...")
            print()
    
    elif args.delete:
        success = processor.delete_document(args.delete)
        print(f"Delete {'successful' if success else 'failed'}")
    
    elif args.stats:
        stats = processor.get_database_stats()
        print(f"Database stats: {stats}")
    
    else:
        print("Please specify --file, --directory, --search, --delete, or --stats")

if __name__ == "__main__":
    main()