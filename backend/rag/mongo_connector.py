"""This module is responsible for the functionality that allows a user to give a mongo db URI and use it to load documents for RAG vector store chunks.

This module provides interactive functionality to:
1. Ask user for database URI
2. Display a list of collections for them to select
3. Ask if they want to join collections or treat them separately
4. Join collections based on IDs if requested
5. Embed the documents for RAG

"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from langchain_core.documents import Document as LC_Document
from langchain.text_splitter import TokenTextSplitter
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from bson import ObjectId
import json

logger = logging.getLogger(__name__)

def convert_bson_to_json(obj: Any) -> Any:
    """
    Convert BSON types to JSON-serializable types.
    
    Args:
        obj: Object that may contain BSON types
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_bson_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_bson_to_json(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

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
            
            # Convert BSON types to JSON-serializable types
            sample_doc_serializable = convert_bson_to_json(sample_doc) if sample_doc else None
            
            return {
                "name": collection_name,
                "document_count": count,
                "sample_document": sample_doc_serializable,
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
            # Convert BSON types to JSON-serializable types
            return [convert_bson_to_json(doc) for doc in docs]
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
        print(f"Loaded {len(raw_docs)} documents from {collection_name}")
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

    def join_collections_by_id(self, collections: List[str], join_field: str = "_id",
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Join multiple collections based on a common field (usually ID).
        
        Args:
            collections: List of collection names to join
            join_field: Field to join on (default: "_id")
            limit: Maximum number of documents to process
            
        Returns:
            List of joined documents
        """
        if len(collections) < 2:
            logger.warning("Need at least 2 collections to join")
            return []
            
        try:
            # Start with the first collection as the base
            base_collection = self.db[collections[0]]
            base_docs = list(base_collection.find({}).limit(limit))
            
            joined_docs = []
            
            for base_doc in base_docs:
                joined_doc = {"_base_collection": collections[0]}
                joined_doc.update(convert_bson_to_json(base_doc))
                
                # Get the join key value
                join_value = base_doc.get(join_field)
                if join_value is None:
                    continue
                
                # Join with other collections
                for collection_name in collections[1:]:
                    other_collection = self.db[collection_name]
                    other_doc = other_collection.find_one({join_field: join_value})
                    
                    if other_doc:
                        other_doc_converted = convert_bson_to_json(other_doc)
                        # Prefix fields with collection name to avoid conflicts
                        for key, value in other_doc_converted.items():
                            if key != join_field:  # Don't duplicate the join field
                                prefixed_key = f"{collection_name}_{key}"
                                joined_doc[prefixed_key] = value
                
                joined_docs.append(joined_doc)
            
            logger.info(f"Successfully joined {len(joined_docs)} documents from collections: {collections}")
            return joined_docs
            
        except Exception as e:
            logger.error(f"Error joining collections: {e}")
            return []

    def load_joined_collections_as_langchain(self, collections: List[str], 
                                           join_field: str = "_id",
                                           content_fields: Optional[List[str]] = None,
                                           limit: int = 100) -> List[LC_Document]:
        """
        Join collections and convert to LangChain Document format.
        
        Args:
            collections: List of collection names to join
            join_field: Field to join on
            content_fields: Fields to use as content (if None, uses all text fields)
            limit: Maximum number of documents
        """
        joined_docs = self.join_collections_by_id(collections, join_field, limit)
        langchain_docs = []
        
        for doc in joined_docs:
            # Extract content from specified fields or all text fields
            if content_fields:
                content_parts = []
                for field in content_fields:
                    if field in doc and doc[field]:
                        content_parts.append(f"{field}: {str(doc[field])}")
                content = "\n".join(content_parts)
            else:
                # Use all non-ID fields as content
                flattened = flatten_dict(doc)
                content_parts = []
                for key, value in flattened.items():
                    if not key.endswith("_id") and value:
                        content_parts.append(f"{key}: {str(value)}")
                content = "\n".join(content_parts)
            
            # Create metadata
            metadata = {
                "source": "mongodb_joined",
                "collections": "_".join(collections),
                "join_field": join_field,
                "doc_id": str(doc.get("_id", "")),
                "loaded_at": datetime.now().isoformat()
            }
            
            # Add non-content fields as metadata
            for key, value in doc.items():
                if key not in ["_id"] and value is not None:
                    if len(str(value)) < 200:  # Only short values as metadata
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


def interactive_mongodb_setup() -> Tuple[str, str, List[str], bool, str]:
    """
    Interactive setup for MongoDB connection and collection selection.
    
    Returns:
        Tuple of (uri, db_name, selected_collections, should_join, join_field)
    """
    print("=== MongoDB RAG Document Loader ===")
    
    # Get database URI
    uri = input("Enter MongoDB URI (e.g., mongodb://localhost:27017): ").strip()
    if not uri:
        uri = "mongodb://localhost:27017"
        print(f"Using default URI: {uri}")
    
    # Get database name
    db_name = input("Enter database name: ").strip()
    if not db_name:
        raise ValueError("Database name is required")
    
    # Test connection and get collections
    try:
        connector = MongoDBConnector(uri, db_name)
        collections = connector.get_collections()
        
        if not collections:
            print("No collections found in the database.")
            connector.close()
            return uri, db_name, [], False, ""
        
        print(f"\nFound {len(collections)} collections:")
        for i, collection in enumerate(collections, 1):
            info = connector.get_collection_info(collection)
            print(f"{i}. {collection} ({info.get('document_count', 0)} documents)")
            if info.get('fields'):
                print(f"   Fields: {', '.join(info['fields'][:5])}{'...' if len(info['fields']) > 5 else ''}")
        
        # Let user select collections
        print("\nSelect collections to use (enter numbers separated by commas, e.g., 1,3,5):")
        selection = input("Selection: ").strip()
        
        try:
            selected_indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_collections = [collections[i] for i in selected_indices if 0 <= i < len(collections)]
        except (ValueError, IndexError):
            print("Invalid selection. Using first collection.")
            selected_collections = [collections[0]]
        
        print(f"Selected collections: {selected_collections}")
        
        # Ask about joining if multiple collections selected
        should_join = False
        join_field = "_id"
        
        if len(selected_collections) > 1:
            join_choice = input("\nDo you want to JOIN these collections or treat them SEPARATELY? (join/separate): ").strip().lower()
            should_join = join_choice.startswith('j')
            
            if should_join:
                join_field = input("Enter field to join on (press Enter for '_id'): ").strip()
                if not join_field:
                    join_field = "_id"
                print(f"Will join collections on field: {join_field}")
            else:
                print("Will treat collections separately")
        
        connector.close()
        return uri, db_name, selected_collections, should_join, join_field
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise


def load_mongodb_documents_interactive() -> List[LC_Document]:
    """
    Interactive function to load MongoDB documents for RAG according to docstring requirements.
    
    This implements the full workflow described in the module docstring:
    1. Ask user for database URI
    2. Display collections for selection
    3. Ask about joining vs separate treatment
    4. Join based on IDs if requested
    5. Return documents ready for embedding
    """
    try:
        # Interactive setup
        uri, db_name, collections, should_join, join_field = interactive_mongodb_setup()
        
        if not collections:
            print("No collections selected.")
            return []
        
        # Load documents
        connector = MongoDBConnector(uri, db_name)
        all_docs = []
        
        try:
            if should_join and len(collections) > 1:
                # Join collections
                print(f"\nJoining collections on '{join_field}'...")
                docs = connector.load_joined_collections_as_langchain(
                    collections=collections,
                    join_field=join_field,
                    limit=1000  # Configurable limit
                )
                all_docs.extend(docs)
                print(f"Loaded {len(docs)} joined documents")
                
            else:
                # Treat collections separately
                print(f"\nLoading collections separately...")
                for collection in collections:
                    print(f"Loading from {collection}...")
                    docs = connector.load_documents_as_langchain(
                        collection_name=collection,
                        limit=1000  # Configurable limit
                    )
                    all_docs.extend(docs)
                    print(f"Loaded {len(docs)} documents from {collection}")
            
            print(f"\nTotal documents loaded: {len(all_docs)}")
            return all_docs
            
        finally:
            connector.close()
            
    except Exception as e:
        logger.error(f"Error in interactive MongoDB loading: {e}")
        print(f"Error: {e}")
        return []


# Legacy function for backward compatibility
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


# Example usage
if __name__ == "__main__":
    # Run the interactive setup
    documents = load_mongodb_documents_interactive()
    print(f"\nReady to embed {len(documents)} documents!")
    
    # Example of what you might do next:
    # from your_embedding_module import embed_documents
    # embedded_docs = embed_documents(documents)
