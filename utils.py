import os
import time
import logging
import torch
import numpy as np
from typing import Dict, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from rembg import remove
from pinecone import Pinecone, ServerlessSpec

# Configure logger
logger = logging.getLogger("image-match")

# Model constants
CLIP_MODEL_ID = "zer0int/LongCLIP-GmP-ViT-L-14"
MAX_TOKEN_LENGTH = 248

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
    """Load LongCLIP model and processor, with caching to avoid reloading"""
    global _clip_model, _clip_processor
    
    # Return cached model if available
    if _clip_model is not None and _clip_processor is not None:
        logger.info("Using cached LongCLIP model")
        return _clip_model, _clip_processor
    
    logger.info(f"Loading LongCLIP model: {CLIP_MODEL_ID}")
    start_time = time.time()
    
    # Implementing LongCLIP with extended token context (248 tokens)
    config = CLIPConfig.from_pretrained(CLIP_MODEL_ID)
    config.text_config.max_position_embeddings = MAX_TOKEN_LENGTH
    
    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, config=config)
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, padding="max_length", max_length=MAX_TOKEN_LENGTH)
    
    logger.info(f"LongCLIP model loaded in {time.time() - start_time:.2f} seconds")
    logger.info(f"Model supports up to {MAX_TOKEN_LENGTH} tokens for text input")
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
    """Generate image and/or text embeddings using LongCLIP"""
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
    
    # Process text - now handling longer text with LongCLIP (up to 248 tokens)
    if text is not None:
        logger.info(f"Generating text embedding for: '{text}'")
        start_time = time.time()
        
        # Using the max_length parameter to properly handle long text
        inputs = processor(text=[text], return_tensors="pt", padding="max_length", max_length=MAX_TOKEN_LENGTH, truncation=True)
        
        # Log token count for debugging
        token_count = len(inputs.input_ids[0])
        if token_count >= MAX_TOKEN_LENGTH:
            logger.warning(f"Text was truncated: {token_count} tokens (max: {MAX_TOKEN_LENGTH})")
        else:
            logger.info(f"Text token count: {token_count} (max: {MAX_TOKEN_LENGTH})")
            
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
                dimension=768,  # LongCLIP's embedding size
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