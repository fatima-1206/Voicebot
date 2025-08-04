from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID

class MongoConnectionRequest(BaseModel):
    uri: str = Field(..., description="MongoDB connection URI")
    db_name: str = Field(..., description="Database name")
    timeout: int = Field(default=5000, description="Connection timeout in milliseconds")

class CollectionInfo(BaseModel):
    name: str
    document_count: int
    fields: List[str]
    sample_document: Optional[Dict[str, Any]] = None

class MongoConnectionResponse(BaseModel):
    success: bool
    message: str
    collections: List[CollectionInfo] = []

class CollectionSelectionRequest(BaseModel):
    uri: str
    db_name: str
    selected_collections: List[str]
    should_join: bool = False
    join_field: str = "_id"
    limit: int = Field(default=1000, description="Maximum documents to load")
    content_fields: Optional[List[str]] = None

class DocumentLoadResponse(BaseModel):
    success: bool
    message: str
    documents_loaded: int
    collections_processed: List[str]
    processing_time: float
    job_id: Optional[str] = None  # For async processing

class MongoDocumentProcessingRequest(BaseModel):
    connection_request: MongoConnectionRequest
    selection_request: CollectionSelectionRequest
    process_async: bool = Field(default=True, description="Process documents asynchronously")

class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: float = 0.0
    message: str = ""
    documents_processed: int = 0
    total_documents: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None