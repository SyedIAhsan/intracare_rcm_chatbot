import os
import logging
from typing import List, Dict, Any
import hashlib

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

logger = logging.getLogger(__name__)

class PineconeHandler:
    def __init__(self, api_key: str = None, environment: str = None, index_name: str = "chatbot-docs"):
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.environment = environment or os.getenv('PINECONE_ENVIRONMENT')
        self.index_name = index_name
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        self.pc = Pinecone(api_key=self.api_key)
        
        if index_name not in self.pc.list_indexes().names():
            self._create_index()
        
        self.index = self.pc.Index(index_name)
    
    def _create_index(self):
        self.pc.create_index(
            name=self.index_name,
            dimension=1536,  # OpenAI ada-002 embedding dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=self.environment or 'us-east-1'
            )
        )
        logger.info(f"Created Pinecone index: {self.index_name}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def _generate_id(self, content: str, metadata: Dict[str, Any]) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        source = metadata.get('source', 'unknown')
        chunk_id = metadata.get('chunk_id', 0)
        return f"{source}_{chunk_id}_{content_hash[:8]}"
    
    def upsert_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, int]:
        vectors = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc['metadata']
            
            if not content.strip():
                continue
            
            try:
                embedding = self._generate_embedding(content)
                vector_id = self._generate_id(content, metadata)
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        **metadata,
                        'content': content[:1000]  # Store first 1000 chars in metadata
                    }
                })
                
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                continue
        
        upserted_count = 0
        failed_count = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
                upserted_count += len(batch)
                logger.info(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
            except Exception as e:
                failed_count += len(batch)
                logger.error(f"Failed to upsert batch {i//batch_size + 1}: {str(e)}")
        
        return {
            'upserted': upserted_count,
            'failed': failed_count,
            'total': len(vectors)
        }
    
    def search(self, query: str, top_k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        query_embedding = self._generate_embedding(query)
        
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        results = []
        for match in search_results['matches']:
            results.append({
                'content': match['metadata'].get('content', ''),
                'score': match['score'],
                'metadata': match['metadata'],
                'id': match['id']
            })
        
        return results
    
    def delete_by_source(self, source_path: str) -> bool:
        try:
            self.index.delete(filter={'source': source_path})
            logger.info(f"Deleted vectors for source: {source_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors for {source_path}: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        return self.index.describe_index_stats()