"""This module is responsible for the functionality that allows a user to give a mongo db URI and use it to load documents for RAG vector store chunks."""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from langchain_core.documents import Document as LC_Document
from langchain.text_splitter import TokenTextSplitter
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MongoDBConnector:
    def __init__(self, uri: str, db_name: str, timeout: int = 5000):
        """
        Initialize MongoDB connector.
        
        Args:
            uri: MongoDB connection URI
            db_name: Database name
            timeout: Connection timeout in milliseconds
        """
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=timeout)
            self.db = self.client[db_name]
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB database: {db_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise e

    def get_collections(self) -> List[str]:
        """Get list of collection names in the database."""
        try:
            return self.db.list_collection_names()
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a specific collection."""
        try:
            collection = self.db[collection_name]
            count = collection.count_documents({})
            sample_doc = collection.find_one({})
            
            return {
                "name": collection_name,
                "document_count": count,
                "sample_document": sample_doc,
                "fields": list(sample_doc.keys()) if sample_doc else []
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def load_documents(self, collection_name: str, limit: int = 100, 
                      query_filter: Optional[Dict] = None,
                      text_fields: Optional[List[str]] = None) -> List[Dict]:
        """
        Load documents from MongoDB collection.
        
        Args:
            collection_name: Name of the collection
            limit: Maximum number of documents to load
            query_filter: MongoDB query filter
            text_fields: Specific fields to extract text from
        """
        try:
            collection = self.db[collection_name]
            query = query_filter or {}
            
            # Use projection if specific text fields are specified
            projection = {field: 1 for field in text_fields} if text_fields else None
            
            docs = collection.find(query, projection).limit(limit)
            return list(docs)
        except Exception as e:
            logger.error(f"Error loading documents from {collection_name}: {e}")
            return []

    def load_documents_as_langchain(self, collection_name: str, limit: int = 100,
                                   query_filter: Optional[Dict] = None,
                                   text_fields: Optional[List[str]] = None,
                                   content_field: str = "content") -> List[LC_Document]:
        """
        Load documents from MongoDB and convert to LangChain Document format.
        
        Args:
            collection_name: Name of the collection
            limit: Maximum number of documents to load
            query_filter: MongoDB query filter
            text_fields: Fields to include in metadata
            content_field: Field to use as main content
        """
        raw_docs = self.load_documents(collection_name, limit, query_filter, text_fields)
        langchain_docs = []
        
        for doc in raw_docs:
            # Extract main content
            if content_field in doc:
                content = str(doc[content_field])
            else:
                # If no specific content field, flatten and use all text
                flattened = flatten_dict(doc)
                content = " ".join([str(v) for v in flattened.values() if v])
            
            # Create metadata (exclude the content field and _id)
            metadata = {
                "source": "mongodb",
                "collection": collection_name,
                "doc_id": str(doc.get("_id", "")),
                "loaded_at": datetime.now().isoformat()
            }
            
            # Add other fields as metadata
            for key, value in doc.items():
                if key not in [content_field, "_id"] and value is not None:
                    metadata[key] = str(value)
            
            langchain_docs.append(LC_Document(
                page_content=content,
                metadata=metadata
            ))
        
        return langchain_docs

    def close(self):
        """Close MongoDB connection."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for nested keys
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Handle lists by joining string representations
            items.append((new_key, " ".join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_docs(docs: List[Dict]) -> List[Dict]:
    """Flatten a list of documents."""
    return [flatten_dict(doc) for doc in docs]

# Example usage function
def load_mongodb_documents_for_rag(uri: str, db_name: str, collection_name: str,
                                  limit: int = 100, content_field: str = "content") -> List[LC_Document]:
    """
    Helper function to load MongoDB documents for RAG system.
    
    Args:
        uri: MongoDB connection URI
        db_name: Database name
        collection_name: Collection name
        limit: Maximum documents to load
        content_field: Field containing main text content
    """
    connector = MongoDBConnector(uri, db_name)
    try:
        docs = connector.load_documents_as_langchain(
            collection_name=collection_name,
            limit=limit,
            content_field=content_field
        )
        logger.info(f"Loaded {len(docs)} documents from MongoDB collection '{collection_name}'")
        return docs
    finally:
        connector.close()
