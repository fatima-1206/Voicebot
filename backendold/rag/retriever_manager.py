"""
Retriever Manager - Handles global coordination of retriever reloading across Celery workers
Uses Redis as a coordination mechanism to signal when retrievers need to be reloaded
"""
import json
import time
from typing import Optional, Tuple, Any
from backend.redis_manager import get_redis_client
from backend.rag import retrieverconstruction
import logging

logger = logging.getLogger(__name__)

# Redis keys for coordination
RETRIEVER_VERSION_KEY = "retriever:version"
RETRIEVER_RELOAD_SIGNAL_KEY = "retriever:reload_signal"
RETRIEVER_LOCK_KEY = "retriever:lock"

class RetrieverManager:
    """Manages retriever state and coordinates reloading across Celery workers"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.current_version: Optional[int] = None
        self.last_check_time = 0
        self.check_interval = 1.0  # Check every 1 second
        
    def get_current_version(self) -> int:
        """Get the current retriever version from Redis"""
        try:
            version = self.redis_client.get(RETRIEVER_VERSION_KEY)
            return int(version) if version else 0
        except Exception as e:
            logger.error(f"Failed to get retriever version: {e}")
            return 0
    
    def increment_version(self) -> int:
        """Increment the retriever version to signal all workers to reload"""
        try:
            new_version = self.redis_client.incr(RETRIEVER_VERSION_KEY)
            # Set reload signal with TTL
            self.redis_client.setex(RETRIEVER_RELOAD_SIGNAL_KEY, 300, str(new_version))  # 5 min TTL
            logger.info(f"🔄 Retriever version incremented to {new_version}")
            return new_version
        except Exception as e:
            logger.error(f"Failed to increment retriever version: {e}")
            return 0
    
    def should_reload_retrievers(self) -> bool:
        """Check if retrievers need to be reloaded based on version changes"""
        current_time = time.time()
        
        # Rate limit checks to avoid constant Redis calls
        if current_time - self.last_check_time < self.check_interval:
            return False
        
        self.last_check_time = current_time
        
        try:
            remote_version = self.get_current_version()
            
            # If this is the first check, store current version
            if self.current_version is None:
                self.current_version = remote_version
                return False
            
            # Check if version has changed
            if remote_version > self.current_version:
                logger.info(f"🔄 Retriever reload needed: version {self.current_version} -> {remote_version}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to check reload status: {e}")
            return False
    
    def mark_retrievers_reloaded(self) -> None:
        """Mark that retrievers have been reloaded by this worker"""
        try:
            self.current_version = self.get_current_version()
            logger.info(f"✅ Retrievers marked as reloaded (version {self.current_version})")
        except Exception as e:
            logger.error(f"Failed to mark retrievers as reloaded: {e}")
    
    def get_retrievers_with_reload_check(self) -> Tuple[Any, Any]:
        """Get retrievers with automatic reload check"""
        try:
            # Check if we need to reload
            if self.should_reload_retrievers():
                logger.info("🔄 Reloading retrievers due to version change...")
                
                # Acquire lock to prevent multiple workers from reloading simultaneously
                lock_acquired = self.redis_client.set(
                    RETRIEVER_LOCK_KEY, 
                    "locked", 
                    nx=True,  # Only set if not exists
                    ex=30     # 30 second expiration
                )
                
                if lock_acquired:
                    try:
                        # Force reload by clearing global state
                        retrieverconstruction.faiss_db = None
                        retrieverconstruction.hybrid_retriever = None
                        retrieverconstruction.db_initialized = False
                        
                        # Get fresh retrievers
                        faiss_db, hybrid_retriever = retrieverconstruction.get_retrievers()
                        
                        # Mark as reloaded
                        self.mark_retrievers_reloaded()
                        
                        logger.info("✅ Retrievers reloaded successfully")
                        return faiss_db, hybrid_retriever
                        
                    finally:
                        # Always release the lock
                        self.redis_client.delete(RETRIEVER_LOCK_KEY)
                else:
                    # Another worker is reloading, wait a bit and get current retrievers
                    time.sleep(0.5)
                    faiss_db, hybrid_retriever = retrieverconstruction.get_retrievers()
                    self.mark_retrievers_reloaded()
                    return faiss_db, hybrid_retriever
            
            # No reload needed, return current retrievers
            return retrieverconstruction.get_retrievers()
            
        except Exception as e:
            logger.error(f"Failed to get retrievers with reload check: {e}")
            # Fallback to normal retriever loading
            return retrieverconstruction.get_retrievers()
    
    def signal_retriever_update(self) -> None:
        """Signal that retrievers have been updated and all workers should reload"""
        self.increment_version()
        logger.info("📡 Signaled all workers to reload retrievers")

# Global instance
retriever_manager = RetrieverManager()

# Convenience functions
def get_retrievers_with_auto_reload() -> Tuple[Any, Any]:
    """Get retrievers with automatic reload check - main entry point for query tasks"""
    return retriever_manager.get_retrievers_with_reload_check()

def signal_retrievers_updated() -> None:
    """Signal that retrievers have been updated - call after adding documents"""
    retriever_manager.signal_retriever_update()