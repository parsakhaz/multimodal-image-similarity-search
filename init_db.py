"""
Initialize ChromaDB for ImageMatch

This script sets up the ChromaDB collection for storing image embeddings
and metadata in the ImageMatch application.

Usage:
    python init_db.py
"""

import os
import time
import logging
import chromadb
import numpy as np
from dotenv import load_dotenv

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("image-match-init")

def init_chromadb():
    """
    Initialize ChromaDB and create the image-match collection if it doesn't exist.
    
    Returns:
        chromadb.Collection: The initialized collection object
    """
    # Load environment variables
    load_dotenv()
    collection_name = os.getenv("COLLECTION_NAME", "image-match")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_data")
    
    # Ensure the persistence directory exists
    logger.info(f"Ensuring ChromaDB persistence directory exists: {persist_dir}")
    os.makedirs(persist_dir, exist_ok=True)
    
    # Initialize ChromaDB client
    logger.info(f"Initializing ChromaDB client with persistence directory: {persist_dir}")
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Create collection if it doesn't exist
    logger.info(f"Creating collection if it doesn't exist: {collection_name}")
    try:
        # Get existing collections - in v0.6.0 this returns only names
        collection_names = client.list_collections()
        logger.info(f"ChromaDB has {len(collection_names)} collection(s)")
        
        # Check if our collection exists
        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' already exists")
            collection = client.get_collection(name=collection_name)
            
            # Get collection count
            count = collection.count()
            logger.info(f"Collection '{collection_name}' contains {count} vectors")
        else:
            # Create new collection with metadata
            logger.info(f"Creating new collection: {collection_name}")
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Using cosine similarity
            )
            logger.info(f"Collection '{collection_name}' created successfully")
        
        return collection
        
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting ChromaDB initialization")
        collection = init_chromadb()
        logger.info("ChromaDB initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {str(e)}") 