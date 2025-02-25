import os
import time
import logging
import torch
import numpy as np
from typing import Dict, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from rembg import remove
from pinecone import Pinecone, ServerlessSpec

# Configure logger
logger = logging.getLogger("image-match")

# Model constants
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Pinecone constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
INDEX_NAME = os.getenv("INDEX_NAME", "image-match")

# Cache for models to avoid reloading
_clip_model = None
_clip_processor = None

@torch.no_grad()
def load_clip_model():
    """Load CLIP model and processor, with caching to avoid reloading"""
    global _clip_model, _clip_processor
    
    # Return cached model if available
    if _clip_model is not None and _clip_processor is not None:
        logger.info("Using cached CLIP model")
        return _clip_model, _clip_processor
    
    logger.info(f"Loading CLIP model: {CLIP_MODEL_ID}")
    start_time = time.time()
    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    logger.info(f"CLIP model loaded in {time.time() - start_time:.2f} seconds")
    return _clip_model, _clip_processor

def remove_background(image: Image.Image) -> Image.Image:
    """Remove background using rembg"""
    start_time = time.time()
    logger.info("Removing background from image")
    result = remove(image)
    logger.info(f"Background removal completed in {time.time() - start_time:.2f} seconds")
    return result

@torch.no_grad()
def generate_clip_embedding(
    image: Image.Image = None,
    text: Optional[str] = None,
    model=None,
    processor=None
) -> Dict[str, np.ndarray]:
    """Generate image and/or text embeddings using CLIP"""
    if model is None or processor is None:
        model, processor = load_clip_model()
    
    result = {}
    
    # Process image
    if image is not None:
        logger.info("Generating image embedding")
        start_time = time.time()
        inputs = processor(images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs)
        image_embedding = image_features / image_features.norm(dim=1, keepdim=True)
        result["image"] = image_embedding.cpu().numpy()
        logger.info(f"Image embedding generated in {time.time() - start_time:.2f} seconds")
    
    # Process text
    if text is not None:
        logger.info(f"Generating text embedding for: '{text}'")
        start_time = time.time()
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        text_features = model.get_text_features(**inputs)
        text_embedding = text_features / text_features.norm(dim=1, keepdim=True)
        result["text"] = text_embedding.cpu().numpy()
        logger.info(f"Text embedding generated in {time.time() - start_time:.2f} seconds")
    
    return result

def init_pinecone():
    """Initialize Pinecone connection and ensure index exists"""
    logger.info("Initializing Pinecone connection...")
    if not PINECONE_API_KEY:
        logger.error("Pinecone API key not found in environment variables")
        raise ValueError("Pinecone API key must be set in .env file")
    
    # Initialize connection
    logger.info("Creating Pinecone client")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    try:
        # Try to get the index
        logger.info(f"Attempting to connect to existing index: {INDEX_NAME}")
        index = pc.Index(INDEX_NAME)
        logger.info(f"Successfully connected to existing index: {INDEX_NAME}")
    except Exception as e:
        # If the index doesn't exist, create it
        logger.info(f"Index not found, creating new index: {INDEX_NAME}")
        logger.info(f"Creating index with dimensions=512, metric=cosine, cloud={PINECONE_CLOUD}, region={PINECONE_REGION}")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=512,  # CLIP's embedding size
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            logger.info("Index creation request sent, waiting for index to be ready...")
            # Check if index exists after creation
            index = pc.Index(INDEX_NAME)
            logger.info(f"New index created and ready: {INDEX_NAME}")
        except Exception as create_error:
            logger.error(f"Error creating index: {create_error}")
            raise
    
    return index 