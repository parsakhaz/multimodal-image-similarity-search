# ImageMatch MVP: A Simple Image Similarity Search Tool
# Uses CLIP embeddings and ChromaDB for storage and search

import os
import uuid
import base64
import logging
import time
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import imagehash

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("image-match")

logger.info("Starting ImageMatch application")
logger.info("Importing dependencies...")

try:
    import numpy as np
    from PIL import Image
    import torch
    from fastapi import FastAPI, File, Form, UploadFile, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    from dotenv import load_dotenv
    
    # Load environment variables FIRST - before any API key access
    logger.info("Loading environment variables...")
    load_dotenv()
    logger.info("Environment variables loaded")
    
    # Import AVIF plugin for Pillow
    try:
        import pillow_avif
        logging.info("AVIF image support enabled via pillow-avif-plugin")
    except ImportError:
        logging.warning("pillow-avif-plugin not found. AVIF image support may be limited.")
    
    # Initialize Moondream AFTER environment variables are loaded
    try:
        import moondream as md
        logging.info("Moondream image captioning enabled")
        MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")
        moondream_model = None
        if MOONDREAM_API_KEY:
            try:
                moondream_model = md.vl(api_key=MOONDREAM_API_KEY)
                logging.info(f"Moondream model initialized successfully with API key: {MOONDREAM_API_KEY[:5]}...")
            except Exception as e:
                logging.error(f"Failed to initialize Moondream model: {e}")
        else:
            logging.warning("MOONDREAM_API_KEY not set. Image captioning will be disabled.")
    except ImportError:
        logging.warning("Moondream package not found. Image captioning will be disabled.")
        moondream_model = None
    
    # Import utility functions after environment variables are loaded
    from utils import (
        load_clip_model,
        remove_background,
        generate_clip_embedding,
        init_chromadb,
        CLIP_MODEL_ID,
        COLLECTION_NAME,
        CHROMA_PERSIST_DIR,
        MAX_TOKEN_LENGTH
    )
    
    logger.info("All dependencies imported successfully")
except Exception as e:
    logger.error(f"Error importing dependencies: {e}")
    raise

# Initialize FastAPI
app = FastAPI(title="ImageMatch MVP")

# Set up directories
logger.info("Setting up directories...")
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/processed", exist_ok=True)
os.makedirs("static/encoded", exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
logger.info("Directories setup complete")

logger.info(f"Using ChromaDB with persistence directory: {CHROMA_PERSIST_DIR}, Collection name: {COLLECTION_NAME}")

# Global ChromaDB collection
collection = None

# Image metadata storage (in-memory for simplicity)
# In a production app, you might want to use a proper database for this
image_metadata = {}

# Load existing metadata from ChromaDB
def load_metadata_from_chromadb():
    """Load all image metadata from ChromaDB to initialize our in-memory cache"""
    global collection
    try:
        logger.info("Loading existing metadata from ChromaDB...")
        # Get all IDs stored in the collection
        all_ids = collection.get(include=[])["ids"]
        
        if not all_ids:
            logger.info("No existing metadata found in ChromaDB.")
            return
            
        # Fetch all metadata for the ids
        results = collection.get(
            ids=all_ids,
            include=["metadatas"]
        )
        
        # Extract metadata
        if results and "metadatas" in results and results["metadatas"]:
            for idx, image_id in enumerate(results["ids"]):
                metadata = results["metadatas"][idx]
                image_metadata[image_id] = metadata
            
        logger.info(f"Loaded metadata for {len(image_metadata)} existing images")
    except Exception as e:
        logger.error(f"Failed to load metadata from ChromaDB: {e}")
        logger.info("Starting with empty metadata cache")

# Generate a deterministic ID for the image based on its content
def generate_image_hash(image: Image.Image) -> str:
    """Generate a perceptual hash of the image to uniquely identify it"""
    # Calculate the perceptual hash of the image
    phash = str(imagehash.phash(image))
    logger.info(f"Generated perceptual hash for image: {phash}")
    return phash

# Generate image caption using Moondream model
def generate_image_caption(image: Image.Image) -> Tuple[Optional[str], Optional[Any]]:
    """Generate a descriptive caption for the image using Moondream
    
    Returns:
        Tuple containing (caption, encoded_image)
    """
    global moondream_model
    
    # Skip if Moondream is not available
    if not moondream_model:
        logger.warning("Image captioning skipped - Moondream not available")
        return None, None
    
    try:
        logger.info("Generating image caption with Moondream")
        start_time = time.time()
        
        # First encode the image
        encoded_image = moondream_model.encode_image(image)
        
        # Generate caption using Moondream with the encoded image
        caption_result = moondream_model.caption(encoded_image)
        caption = caption_result["caption"]
        
        logger.info(f"Caption generated in {time.time() - start_time:.2f} seconds: {caption}")
        return caption, encoded_image
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return None, None

# Process and store image
def process_image(
    image: Image.Image,
    filename: str,
    description: Optional[str] = None,
    custom_metadata: Optional[str] = None
) -> Tuple[Dict, bool]:
    """Process image and store in ChromaDB
    
    Returns:
        Tuple containing (metadata, is_new_upload)
    """
    # Generate content-based ID
    image_id = generate_image_hash(image)
    
    # Check if this image already exists in our system
    existing_check = collection.get(
        ids=[image_id],
        include=["metadatas"]
    )
    
    # If the image already exists, return its metadata
    if existing_check and existing_check["ids"]:
        logger.info(f"Image with hash {image_id} already exists, skipping processing")
        metadata_idx = existing_check["ids"].index(image_id)
        return existing_check["metadatas"][metadata_idx], False
    
    # Generate caption for the image using Moondream
    generated_caption, encoded_image = generate_image_caption(image)
    logger.info(f"Generated caption: {generated_caption}")
    
    # Save encoded image if available
    if encoded_image is not None:
        import torch
        encoded_path = f"static/encoded/{image_id}.pt"
        torch.save(encoded_image, encoded_path)
        logger.info(f"Encoded image saved to {encoded_path}")
    
    # Remove background
    try:
        clean_image = remove_background(image)
    except Exception as e:
        logger.error(f"Background removal error: {e}")
        clean_image = image
    
    # Save processed image
    processed_path = f"static/processed/{image_id}.png"
    clean_image.save(processed_path)
    logger.info(f"Processed image saved to {processed_path}")
    
    # Only use description provided by user, don't fall back to generated caption
    if not description:
        # Fall back to filename as simple description if no description provided
        description = f"An image of {os.path.splitext(filename)[0]}"
    
    # Prepare custom metadata, including AI caption if available
    processed_custom_metadata = custom_metadata or ""
    if generated_caption:
        # Add a separator if there's existing custom metadata
        if processed_custom_metadata:
            processed_custom_metadata += "\n\n"
        # Add the caption text with a label
        processed_custom_metadata += f"{generated_caption}"
        logger.info(f"Added AI caption to custom metadata: {generated_caption}")
    
    # Generate CLIP embedding
    embeddings = generate_clip_embedding(clean_image)
    clip_embedding = embeddings["image"][0].tolist()
    
    # Store metadata
    metadata = {
        "id": image_id,
        "original_filename": filename,
        "processed_path": processed_path,
        "description": description,
        "ai_caption": "",  # Empty string instead of None
        "custom_metadata": processed_custom_metadata,
        "upload_time": datetime.now().isoformat()
    }
    
    # Store in ChromaDB
    logger.info(f"Storing image {image_id} in ChromaDB collection")
    collection.add(
        ids=[image_id],
        embeddings=[clip_embedding],
        metadatas=[metadata],
        documents=[description]  # Store description as document for text search
    )
    logger.info(f"Image {image_id} stored successfully with metadata: {metadata}")
    
    # Store metadata locally for quick access
    image_metadata[image_id] = metadata
    
    return metadata, True

# Search for similar images
def search_similar(
    embedding: np.ndarray,
    limit: int = 10
) -> List[Dict]:
    """Search for similar images by vector similarity using ChromaDB"""
    logger.info(f"Searching for similar images (limit: {limit})")
    
    # Query the collection
    start_time = time.time()
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=limit,
        include=["metadatas", "distances"]
    )
    logger.info(f"Search completed in {time.time() - start_time:.2f} seconds")
    
    # Format results
    formatted_results = []
    if results and "distances" in results and results["distances"]:
        for idx, distance in enumerate(results["distances"][0]):
            # Check if metadatas exists and is properly indexed
            if "metadatas" in results and results["metadatas"] and len(results["metadatas"][0]) > idx:
                # ChromaDB returns metadatas as a list of lists, access it correctly
                result = results["metadatas"][0][idx].copy()
                # ChromaDB returns distance (lower is better) so convert to similarity score (higher is better)
                result["similarity"] = 1.0 - float(distance)
                formatted_results.append(result)
    
    logger.info(f"Found {len(formatted_results)} similar images")
    return formatted_results

# Text search for images
def search_by_text(
    query_text: str,
    limit: int = 10
) -> List[Dict]:
    """Search for images using text query"""
    logger.info(f"Text search with query: '{query_text}'")
    
    # Option 1: Use ChromaDB's text search capabilities
    try:
        # Try to use ChromaDB's native text search
        results = collection.query(
            query_texts=[query_text],
            n_results=limit,
            include=["metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and "distances" in results and results["distances"]:
            for idx, distance in enumerate(results["distances"][0]):
                if idx < len(results["metadatas"]) and results["metadatas"][idx]:
                    result = results["metadatas"][idx].copy()
                    # ChromaDB returns distance (lower is better) so convert to similarity score (higher is better)
                    result["similarity"] = 1.0 - float(distance)
                    formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} images via ChromaDB text search")
        return formatted_results
        
    except Exception as e:
        # If native text search fails, fall back to embedding-based search
        logger.warning(f"Native text search failed: {e}. Falling back to embedding-based search.")
        
        # Option 2: Generate text embedding and search by vector
        # Generate text embedding
        embeddings = generate_clip_embedding(text=query_text)
        text_embedding = embeddings["text"][0]
        
        # Search using vector similarity
        return search_similar(text_embedding, limit)

# Multimodal search combining text and image
def search_multimodal(
    image: Image.Image,
    query_text: str,
    weight_image: float = 0.5,
    limit: int = 10
) -> List[Dict]:
    """Search for images using both image and text queries
    
    Args:
        image: The image to search with
        query_text: The text to search with
        weight_image: Weight given to the image embedding (0-1)
                     Higher values prioritize image similarity
                     Lower values prioritize text similarity
        limit: Maximum number of results to return
    
    Returns:
        List of matching images with metadata and similarity scores
    """
    logger.info(f"Multimodal search with image and text: '{query_text}'")
    
    # Process image for better results (optional)
    try:
        clean_image = remove_background(image)
    except Exception as e:
        logger.warning(f"Background removal failed, using original image: {str(e)}")
        clean_image = image
    
    # Generate embeddings for both modalities
    image_embeddings = generate_clip_embedding(clean_image)
    image_embedding = image_embeddings["image"][0]
    
    text_embeddings = generate_clip_embedding(text=query_text)
    text_embedding = text_embeddings["text"][0]
    
    # Calculate weight for text embedding
    weight_text = 1.0 - weight_image
    
    # Combine embeddings by weighted average
    combined_embedding = (image_embedding * weight_image) + (text_embedding * weight_text)
    
    # Normalize the combined embedding to unit length
    combined_norm = np.linalg.norm(combined_embedding)
    if combined_norm > 0:
        combined_embedding = combined_embedding / combined_norm
    
    logger.info(f"Combined multimodal embedding with weights: image={weight_image:.2f}, text={weight_text:.2f}")
    
    # Search using the combined embedding
    return search_similar(combined_embedding, limit)

# Load encoded image from disk
def load_encoded_image(image_id: str) -> Optional[Any]:
    """Load the encoded image from disk if it exists
    
    Args:
        image_id: The unique ID of the image
        
    Returns:
        The encoded image tensor or None if not found
    """
    try:
        import torch
        encoded_path = f"static/encoded/{image_id}.pt"
        if os.path.exists(encoded_path):
            logger.info(f"Loading encoded image from {encoded_path}")
            return torch.load(encoded_path)
        else:
            logger.warning(f"Encoded image not found for {image_id}")
            return None
    except Exception as e:
        logger.error(f"Error loading encoded image: {e}")
        return None

# Clear all data (reset function)
def reset_system():
    """Reset the entire system by clearing ChromaDB collection and processed images"""
    logger.info("Resetting system - clearing all data")
    
    # Clear ChromaDB collection
    try:
        global collection
        collection.delete(ids=collection.get(include=[])["ids"])
        
        logger.info(f"Deleted {len(collection.get(include=[])['ids'])} vectors from ChromaDB collection")
    except Exception as e:
        logger.error(f"Error clearing ChromaDB collection: {e}")
        raise
    
    # Clear processed images directory
    try:
        processed_dir = "static/processed"
        count = 0
        for filename in os.listdir(processed_dir):
            file_path = os.path.join(processed_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                count += 1
        logger.info(f"Deleted {count} processed image files")
        
        # Clear encoded images directory
        encoded_dir = "static/encoded"
        encoded_count = 0
        for filename in os.listdir(encoded_dir):
            file_path = os.path.join(encoded_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                encoded_count += 1
        logger.info(f"Deleted {encoded_count} encoded image files")
    except Exception as e:
        logger.error(f"Error clearing processed or encoded images: {e}")
        raise
    
    # Clear in-memory metadata cache
    global image_metadata
    image_metadata = {}
    logger.info("Cleared in-memory metadata cache")
    
    return {"success": True, "message": f"System reset complete. Deleted {len(collection.get(include=[])['ids']) if collection.get(include=[])['ids'] else 0} vectors and {count if 'count' in locals() else 0} image files."}

# API Routes
@app.get("/", response_class=HTMLResponse)
def home():
    """Simple HTML interface"""
    logger.info("Home page accessed")
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ImageMatch MVP</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            img { max-width: 150px; max-height: 150px; margin: 5px; border: 1px solid #ddd; }
            .result-item { margin-bottom: 20px; }
            input, textarea, button { margin: 10px 0; }
            textarea { width: 100%; height: 100px; }
            .search-options { display: flex; gap: 20px; }
            .search-box { flex: 1; }
            .highlight { background-color: #f0f7ff; border-left: 4px solid #0066cc; padding-left: 15px; }
            .new-feature { background-color: #f0fff0; border-left: 4px solid #00cc66; padding-left: 15px; }
            .ai-feature { background-color: #e6f7ff; border-left: 4px solid #0099ff; padding-left: 15px; }
        </style>
    </head>
    <body>
        <h1>ImageMatch</h1>
        
        <div class="section ai-feature">
            <h2>Automatic Image Captioning</h2>
            <p>We automatically generate descriptive captions for your images using Moondream for better search results.</p>
        </div>
        
        <div class="section">
            <h2>Upload Image</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br>
                <input type="text" name="description" placeholder="Brief description (optional - AI will generate one if empty)"><br>
                <textarea name="custom_metadata" placeholder="Custom metadata (JSON, keywords, or free text)"></textarea><br>
                <button type="submit">Upload</button>
            </form>
        </div>
        
        <div class="section highlight">
            <h2>Search Options</h2>
            <p>ImageMatch supports multiple search methods:</p>
            
            <div class="search-options">
                <div class="search-box">
                    <h3>Search by Image</h3>
                    <p>Upload an image to find visually similar images</p>
                    <form action="/search/image" method="post" enctype="multipart/form-data">
                        <input type="file" name="file" accept="image/*" required><br>
                        <button type="submit">Search by Image</button>
                    </form>
                </div>
                
                <div class="search-box">
                    <h3>Search by Text</h3>
                    <p>Enter keywords to find semantically matching images</p>
                    <form action="/search/text" method="get">
                        <input type="text" name="query" placeholder="e.g., red drill, cat, landscape" required><br>
                        <button type="submit">Search by Text</button>
                    </form>
                </div>
            </div>
        </div>
        
        <div class="section new-feature">
            <h2>Multimodal Search</h2>
            <p>Combine the power of both <strong>image and text</strong> to find exactly what you're looking for.</p>
            
            <form action="/search/multimodal" method="post" enctype="multipart/form-data">
                <div class="search-options">
                    <div class="search-box">
                        <h3>Image Input</h3>
                        <input type="file" name="file" accept="image/*" required>
                    </div>
                    
                    <div class="search-box">
                        <h3>Text Input</h3>
                        <input type="text" name="query" placeholder="Describe what you're looking for" required>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <label for="weight_image">Balance between image and text influence:</label><br>
                    <div style="display: flex; align-items: center; margin-top: 10px;">
                        <span>Text</span>
                        <input type="range" id="weight_image" name="weight_image" min="0" max="1" step="0.1" value="0.5" style="margin: 0 10px; width: 200px;">
                        <span>Image</span>
                    </div>
                </div>
                
                <button type="submit" style="margin-top: 20px; padding: 10px 20px; background-color: #00cc66; color: white; border: none; border-radius: 4px;">Search with Both</button>
            </form>
        </div>
        
        <div class="section">
            <h2>Manage Database</h2>
            <a href="/images">View all stored images and metadata</a>
        </div>
        
        <div class="section">
            <h2>Admin</h2>
            <a href="/reset-confirm" style="color: red;">Reset System (Clear All Data)</a>
            <p style="font-size: 0.8em;">Warning: This will delete all images and metadata</p>
        </div>
        
        <div id="results" class="section">
            <h2>Getting Started</h2>
            <p>To quickly populate the system with sample images:</p>
            <p><a href="/upload-samples">Click here to upload all sample images from the /images directory</a></p>
        </div>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    description: str = Form(None),
    custom_metadata: str = Form(None)
):
    """Upload and process an image"""
    logger.info(f"Upload request received for file: {file.filename}")
    
    # Read file content
    content = await file.read()
    
    # Handle image opening with better error handling
    try:
        # Try opening the image directly
        image = Image.open(BytesIO(content))
        logger.info(f"Image opened successfully: {file.filename}")
        
        # Convert to RGB if needed (for RGBA, CMYK or other color modes)
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB mode")
            image = image.convert('RGB')
            
    except Exception as e:
        logger.error(f"Error opening image: {str(e)}")
        
        # Save to temporary file and try a different approach
        try:
            logger.info(f"Attempting alternative method for problematic format")
            temp_path = f"static/uploads/temp_{os.path.basename(file.filename)}"
            with open(temp_path, "wb") as f:
                f.write(content)
                
            # Try using a different approach to open the file
            image = Image.open(temp_path)
            logger.info(f"Image opened successfully with alternative method")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Clean up temp file
            os.remove(temp_path)
            
        except Exception as e2:
            logger.error(f"Failed with alternative method: {str(e2)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid or unsupported image format. Please convert your image to JPG, PNG, or other common formats. Error: {str(e)}"
            )
    
    # Save original image
    original_path = f"static/uploads/{file.filename}"
    with open(original_path, "wb") as f:
        f.write(content)
    logger.info(f"Original image saved to {original_path}")
    
    # Process image
    result, is_new_upload = process_image(image, file.filename, description, custom_metadata)
    logger.info(f"Image processed successfully: {result['id']}")
    
    # Different status message based on whether the image was new or a duplicate
    status_message = ""
    if is_new_upload:
        status_message = "<h1>Upload Complete</h1>"
    else:
        status_message = '<h1 style="color: orange;">Duplicate Image Detected</h1><p>This image was already in our database. Using existing metadata and embeddings.</p>'
    
    # Remove the separate AI caption section - it's now only in custom metadata
    ai_caption_section = ""
    
    # Return HTML response
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Result</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .image-container {{ display: flex; margin: 20px 0; }}
            .image-box {{ margin-right: 20px; text-align: center; }}
            img {{ max-width: 300px; max-height: 300px; }}
            a {{ display: block; margin: 20px 0; }}
            .metadata {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 20px; }}
            .duplicate-notice {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid orange; margin-bottom: 20px; }}
            .ai-caption {{ background-color: #e6f7ff; padding: 15px; border-radius: 5px; border-left: 5px solid #0099ff; margin-top: 20px; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; max-width: 100%; }}
        </style>
    </head>
    <body>
        {status_message}
        
        <div class="image-container">
            <div class="image-box">
                <h3>Original Image</h3>
                <img src="/{original_path}" alt="Original image">
            </div>
            <div class="image-box">
                <h3>Processed Image (Background Removed)</h3>
                <img src="/{result['processed_path']}" alt="Processed image">
            </div>
        </div>
        
        <p><strong>Image ID (hash):</strong> {result['id']}</p>
        <p><strong>Description:</strong> {result['description']}</p>
        
        <div class="metadata">
            <h3>Custom Metadata:</h3>
            <pre>{result['custom_metadata'] or 'None provided'}</pre>
        </div>
        
        <a href="/">Back to Home</a>
    </body>
    </html>
    """)

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...)):
    """Search for similar images using an uploaded image"""
    logger.info(f"Image search request received with file: {file.filename}")
    
    # Read file content
    content = await file.read()
    
    # Handle image opening with better error handling
    try:
        # Try opening the image directly
        image = Image.open(BytesIO(content))
        logger.info(f"Search image opened successfully: {file.filename}")
        
        # Convert to RGB if needed (for RGBA, CMYK or other color modes)
        if image.mode != 'RGB':
            logger.info(f"Converting search image from {image.mode} to RGB mode")
            image = image.convert('RGB')
            
    except Exception as e:
        logger.error(f"Error opening search image: {str(e)}")
        
        # Save to temporary file and try a different approach
        try:
            logger.info(f"Attempting alternative method for problematic format")
            temp_path = f"static/uploads/temp_search_{os.path.basename(file.filename)}"
            with open(temp_path, "wb") as f:
                f.write(content)
                
            # Try using a different approach to open the file
            image = Image.open(temp_path)
            logger.info(f"Search image opened successfully with alternative method")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Clean up temp file
            os.remove(temp_path)
            
        except Exception as e2:
            logger.error(f"Failed with alternative method: {str(e2)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid or unsupported image format. Please convert your image to JPG, PNG, or other common formats. Error: {str(e)}"
            )
    
    # Remove background (optional for search)
    try:
        clean_image = remove_background(image)
    except Exception as e:
        logger.warning(f"Background removal failed, using original image: {str(e)}")
        clean_image = image
    
    # Generate embedding
    embeddings = generate_clip_embedding(clean_image)
    
    # Search for similar images
    results = search_similar(embeddings["image"][0])
    
    # Format results for display
    result_html = ""
    for r in results:
        similarity_pct = f"{r['similarity'] * 100:.1f}%"
        result_html += f"""
        <div class="result">
            <img src="/{r['processed_path']}" alt="{r['description']}">
            <p class="similarity">Similarity: {similarity_pct}</p>
            <p>{r['description']}</p>
        </div>
        """
    
    # Return HTML response
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #333; }}
            #results {{ display: flex; flex-wrap: wrap; }}
            .result {{ margin: 10px; text-align: center; width: 220px; border: 1px solid #ddd; padding: 10px; }}
            img {{ max-width: 200px; max-height: 200px; }}
            .similarity {{ font-weight: bold; color: #4CAF50; }}
            a {{ display: block; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Search Results</h1>
        <div id="query-image" style="text-align: center; margin-bottom: 20px;">
            <h3>Query Image</h3>
            <img src="data:image/jpeg;base64,{base64.b64encode(content).decode()}" 
                 style="max-width: 300px; max-height: 300px;">
        </div>
        <div id="results">
            {result_html if result_html else "<p>No similar images found</p>"}
        </div>
        <a href="/">Back to Home</a>
    </body>
    </html>
    """)

@app.get("/search/text")
async def search_by_text_route(query: str):
    """Search for images using text query"""
    logger.info(f"Text search request received with query: '{query}'")
    
    # Search using text
    results = search_by_text(query)
    
    # Format results for display
    result_html = ""
    for r in results:
        similarity_pct = f"{r['similarity'] * 100:.1f}%"
        result_html += f"""
        <div class="result">
            <img src="/{r['processed_path']}" alt="{r['description']}">
            <p class="similarity">Similarity: {similarity_pct}</p>
            <p>{r['description']}</p>
        </div>
        """
    
    # Return HTML response
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text Search Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #333; }}
            #results {{ display: flex; flex-wrap: wrap; }}
            .result {{ margin: 10px; text-align: center; width: 220px; border: 1px solid #ddd; padding: 10px; }}
            img {{ max-width: 200px; max-height: 200px; }}
            .similarity {{ font-weight: bold; color: #4CAF50; }}
            a {{ display: block; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Text Search Results</h1>
        <div id="query" style="margin-bottom: 20px;">
            <h3>Query: "{query}"</h3>
        </div>
        <div id="results">
            {result_html if result_html else "<p>No similar images found</p>"}
        </div>
        <a href="/">Back to Home</a>
    </body>
    </html>
    """)

@app.post("/search/multimodal")
async def search_by_multimodal(
    file: UploadFile = File(...),
    query: str = Form(...),
    weight_image: float = Form(0.5)
):
    """Search for images using both uploaded image and text query"""
    logger.info(f"Multimodal search request received with file: {file.filename}, text: '{query}'")
    
    # Validate weight parameter
    weight_image = min(max(weight_image, 0.0), 1.0)  # Clamp between 0 and 1
    
    # Read file content
    content = await file.read()
    
    # Handle image opening with better error handling
    try:
        # Try opening the image directly
        image = Image.open(BytesIO(content))
        logger.info(f"Search image opened successfully: {file.filename}")
        
        # Convert to RGB if needed (for RGBA, CMYK or other color modes)
        if image.mode != 'RGB':
            logger.info(f"Converting search image from {image.mode} to RGB mode")
            image = image.convert('RGB')
            
    except Exception as e:
        logger.error(f"Error opening search image: {str(e)}")
        
        # Save to temporary file and try a different approach
        try:
            logger.info(f"Attempting alternative method for problematic format")
            temp_path = f"static/uploads/temp_search_{os.path.basename(file.filename)}"
            with open(temp_path, "wb") as f:
                f.write(content)
                
            # Try using a different approach to open the file
            image = Image.open(temp_path)
            logger.info(f"Search image opened successfully with alternative method")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Clean up temp file
            os.remove(temp_path)
            
        except Exception as e2:
            logger.error(f"Failed with alternative method: {str(e2)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid or unsupported image format. Please convert your image to JPG, PNG, or other common formats. Error: {str(e)}"
            )
    
    # Generate caption for the search image using Moondream
    generated_caption, encoded_image = generate_image_caption(image)
    
    # Store full caption for display purposes
    full_caption = generated_caption if generated_caption else ""
    
    # Create an enhanced query by combining user query with the AI caption
    # LongCLIP has a maximum context length of 248 tokens (upgraded from CLIP's 77)
    
    # MODIFIED APPROACH: Prioritize the user's query over the AI caption
    # If user query is very long, use it alone
    # If user query is short, add as much of the AI caption as will fit
    
    # Using a slightly conservative limit for safety
    CLIP_TOKEN_LIMIT = MAX_TOKEN_LENGTH - 10  # Leaving buffer for padding tokens
    
    # Estimate query length in tokens (rough approximation using character count)
    # A better implementation would use a proper tokenizer, but this is a reasonable approximation
    estimated_query_tokens = len(query) / 4  # Rough estimate: ~4 chars per token on average
    
    enhanced_query = query
    caption_was_added = False
    
    # With LongCLIP's higher token limit, we can now include more of the caption
    if generated_caption:
        # Check if we have space for the caption
        if estimated_query_tokens < CLIP_TOKEN_LIMIT - 5:  # Leave some buffer
            # Calculate approximately how many tokens we have left for the caption
            approx_tokens_left = CLIP_TOKEN_LIMIT - estimated_query_tokens
            approx_chars_left = int(approx_tokens_left * 4)  # Convert back to character estimate
            
            # Truncate caption if needed
            truncated_caption = generated_caption
            if len(generated_caption) > approx_chars_left:
                # Cut to approximate length and try to end at a complete word
                truncated_caption = generated_caption[:approx_chars_left]
                # Try to end at the last complete word
                last_space = truncated_caption.rfind(" ", 0, approx_chars_left)
                if last_space > 0:
                    truncated_caption = generated_caption[:last_space]
            
            # With higher token limits, we can almost always add the caption
            enhanced_query = f"{query} {truncated_caption}"
            caption_was_added = True
            logger.info(f"Enhanced query with AI caption (using LongCLIP's extended token limit): '{enhanced_query}'")
        else:
            logger.info(f"User query takes up most of the token limit. Using only user query: '{query}'")
    else:
        logger.info(f"No AI caption generated. Using only user query: '{query}'")
    
    # Search using both image and enhanced text
    results = search_multimodal(image, enhanced_query, weight_image=weight_image)
    
    # Format results for display
    result_html = ""
    for r in results:
        similarity_pct = f"{r['similarity'] * 100:.1f}%"
        result_html += f"""
        <div class="result">
            <img src="/{r['processed_path']}" alt="{r['description']}">
            <p class="similarity">Similarity: {similarity_pct}</p>
            <p>{r['description']}</p>
        </div>
        """
    
    # Format the weight as a percentage for display
    image_weight_pct = f"{weight_image * 100:.0f}%"
    text_weight_pct = f"{(1 - weight_image) * 100:.0f}%"
    
    # Prepare caption display section
    caption_section = ""
    if full_caption:
        caption_note = "This caption was automatically added to your search query (space permitting)" if caption_was_added else "Your query was prioritized, so this caption was not included in the search"
        caption_section = f"""
        <div class="ai-caption">
            <h4>AI Caption for Your Image:</h4>
            <p><em>"{full_caption}"</em></p>
            <p class="note">{caption_note}</p>
        </div>
        """
    
    # Display the actual query used for embedding generation
    final_query_section = f"""
    <div class="final-query">
        <h4>Final Query Used for Search:</h4>
        <p style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;"><code>{enhanced_query}</code></p>
        <p class="note">This is the exact text used to generate the CLIP embedding for similarity search</p>
    </div>
    """
    
    # Return HTML response
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multimodal Search Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .search-params {{ display: flex; margin-bottom: 30px; background-color: #f0f7ff; padding: 15px; border-radius: 5px; }}
            .image-query {{ flex: 1; text-align: center; }}
            .text-query {{ flex: 1; padding: 15px; }}
            .weights {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 10px; }}
            #results {{ display: flex; flex-wrap: wrap; }}
            .result {{ margin: 10px; text-align: center; width: 220px; border: 1px solid #ddd; padding: 10px; }}
            img {{ max-width: 200px; max-height: 200px; }}
            .similarity {{ font-weight: bold; color: #4CAF50; }}
            a {{ display: block; margin: 20px 0; }}
            .ai-caption {{ background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #0099ff; }}
            .final-query {{ background-color: #fff8e6; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #ffa600; }}
            .note {{ font-size: 0.9em; color: #666; }}
            code {{ word-wrap: break-word; overflow-wrap: break-word; white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <h1>Multimodal Search Results</h1>
        
        <div class="search-params">
            <div class="image-query">
                <h3>Image Query</h3>
                <img src="data:image/jpeg;base64,{base64.b64encode(content).decode()}" 
                     style="max-width: 250px; max-height: 250px;">
            </div>
            
            <div class="text-query">
                <h3>Text Query</h3>
                <p>Original query: "{query}"</p>
                
                {caption_section}
                
                {final_query_section}
                
                <div class="weights">
                    <h4>Search Weights</h4>
                    <p>Image Influence: {image_weight_pct}</p>
                    <p>Text Influence: {text_weight_pct}</p>
                </div>
            </div>
        </div>
        
        <h2>Results</h2>
        <div id="results">
            {result_html if result_html else "<p>No similar images found</p>"}
        </div>
        
        <div>
            <h3>Try Different Weights</h3>
            <form action="/search/multimodal" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <input type="text" name="query" value="{query}" required>
                <label for="weight_image">Image Influence:</label>
                <input type="range" id="weight_image" name="weight_image" min="0" max="1" step="0.1" value="{weight_image}">
                <button type="submit">Search Again</button>
            </form>
        </div>
        
        <a href="/">Back to Home</a>
    </body>
    </html>
    """)

@app.get("/upload-samples")
async def upload_sample_images():
    """Upload all images from the /images directory"""
    images_dir = "images"
    if not os.path.exists(images_dir):
        logger.error(f"Sample images directory not found: {images_dir}")
        return HTMLResponse(f"Sample images directory not found: {images_dir}. Please create this directory and add some sample images.")
    
    # Process all images in the directory
    uploaded_new = []
    uploaded_existing = []
    failed = []
    logger.info(f"Processing images from directory: {images_dir}")
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif', '.bmp', '.tiff', '.gif')):
            try:
                file_path = os.path.join(images_dir, filename)
                logger.info(f"Processing sample image: {file_path}")
                
                # Open image with better error handling
                try:
                    image = Image.open(file_path)
                    
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        logger.info(f"Converting image from {image.mode} to RGB mode")
                        image = image.convert('RGB')
                        
                except Exception as e:
                    logger.error(f"Error opening image {filename}: {str(e)}")
                    failed.append(f"{filename} - Error opening image: {str(e)}")
                    continue
                
                # Add default metadata for sample images
                description = f"Sample image: {os.path.splitext(filename)[0]}"
                custom_metadata = f"Sample image loaded from {file_path}. This is an automatically processed sample."
                
                # AI caption will automatically be added to custom_metadata in the process_image function
                result, is_new_upload = process_image(image, filename, description, custom_metadata)
                
                if is_new_upload:
                    uploaded_new.append(f"{filename} - {result['id']} - Caption: {result.get('ai_caption', 'None')}")
                    logger.info(f"Sample image processed successfully (new): {filename}")
                else:
                    uploaded_existing.append(f"{filename} - {result['id']} (already existed)")
                    logger.info(f"Sample image was already in database: {filename}")
                    
            except Exception as e:
                logger.error(f"Failed to process sample image {filename}: {str(e)}")
                failed.append(f"{filename} - {str(e)}")
    
    # Generate HTML response
    uploaded_new_html = "<br>".join(uploaded_new) if uploaded_new else "No new images were uploaded."
    uploaded_existing_html = "<br>".join(uploaded_existing) if uploaded_existing else "No existing images were found."
    failed_html = "<br>".join(failed) if failed else "No upload failures."
    logger.info(f"Sample upload complete. New: {len(uploaded_new)}, Existing: {len(uploaded_existing)}, Failed: {len(failed)}")
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample Images Upload</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h2 {{ color: #333; margin-top: 20px; }}
            .success {{ color: green; }}
            .existing {{ color: orange; }}
            .error {{ color: red; }}
            a {{ display: block; margin: 20px 0; }}
            .info-box {{ background-color: #f0f7ff; padding: 10px; border-radius: 5px; border-left: 5px solid #0066cc; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Sample Images Upload</h1>
        
        <div class="info-box">
            <p><strong>Note:</strong> AI-generated captions are automatically included in the custom metadata field for better search results.</p>
        </div>
        
        <h2 class="success">Successfully Uploaded New Images:</h2>
        <p>{uploaded_new_html}</p>
        
        <h2 class="existing">Existing Images (Already in Database):</h2>
        <p>{uploaded_existing_html}</p>
        
        <h2 class="error">Failed Uploads:</h2>
        <p>{failed_html}</p>
        
        <a href="/">Back to Home</a>
    </body>
    </html>
    """)

@app.get("/images")
async def view_all_images():
    """View all stored images with their metadata"""
    logger.info("Retrieving all images from the database")
    
    # If metadata cache is empty, try to reload from ChromaDB
    if not image_metadata:
        load_metadata_from_chromadb()
    
    # Generate HTML for each image
    image_items = []
    for image_id, metadata in image_metadata.items():
        # Check if AI caption exists and is meaningful
        ai_caption_html = ""
        if metadata.get('ai_caption') and metadata.get('ai_caption') != "No caption generated":
            ai_caption_html = f"""
            <div class="ai-caption">
                <h4>AI Caption:</h4>
                <p><em>"{metadata.get('ai_caption')}"</em></p>
            </div>
            """
            
        image_html = f"""
        <div class="image-item">
            <h3>{metadata.get('description', 'No description')}</h3>
            <div class="image-details">
                <div class="image-preview">
                    <img src="/{metadata.get('processed_path', '')}" alt="{metadata.get('description', 'Image')}">
                </div>
                <div class="image-metadata">
                    <p><strong>ID:</strong> {image_id}</p>
                    <p><strong>Original filename:</strong> {metadata.get('original_filename', 'Unknown')}</p>
                    <p><strong>Upload time:</strong> {metadata.get('upload_time', 'Unknown')}</p>
                    {ai_caption_html}
                    <div class="custom-metadata">
                        <h4>Custom Metadata:</h4>
                        <pre>{metadata.get('custom_metadata', 'None')}</pre>
                    </div>
                    <div class="actions">
                        <a href="/edit-metadata/{image_id}" class="edit-button">Edit Metadata</a>
                    </div>
                </div>
            </div>
        </div>
        """
        image_items.append(image_html)
    
    image_list_html = "\n".join(image_items)
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>All Images</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .image-item {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .image-details {{ display: flex; }}
            .image-preview {{ margin-right: 20px; }}
            .image-metadata {{ flex-grow: 1; }}
            img {{ max-width: 200px; max-height: 200px; }}
            .custom-metadata {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 10px; }}
            .ai-caption {{ background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 3px solid #0099ff; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; max-width: 100%; overflow-wrap: break-word; }}
            a {{ display: block; margin: 20px 0; }}
            .edit-button {{ 
                display: inline-block;
                background-color: #4CAF50; 
                color: white; 
                padding: 8px 16px; 
                text-decoration: none; 
                border-radius: 4px;
                margin-top: 10px;
            }}
            .edit-button:hover {{ 
                background-color: #45a049; 
            }}
            .actions {{ 
                margin-top: 15px; 
            }}
        </style>
    </head>
    <body>
        <h1>All Stored Images</h1>
        <p>Total images: {len(image_metadata)}</p>
        
        <div class="image-list">
            {image_list_html if image_metadata else "<p>No images found in the database.</p>"}
        </div>
        
        <div style="margin-top: 30px; border-top: 1px solid #ddd; padding-top: 20px;">
            <a href="/">Back to Home</a>
            <a href="/reset-confirm" style="color: red; margin-left: 20px;">Reset System (Clear All Data)</a>
        </div>
    </body>
    </html>
    """)

@app.get("/reset-confirm", response_class=HTMLResponse)
async def reset_confirm():
    """Show confirmation page before resetting the system"""
    logger.info("Reset confirmation page accessed")
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reset System Confirmation</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .warning { color: red; font-weight: bold; }
            .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            button.danger { background-color: #ff4444; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button.cancel { background-color: #888; color: white; padding: 10px 20px; border: none; cursor: pointer; margin-right: 20px; }
            .actions { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>Reset System Confirmation</h1>
        
        <div class="section">
            <h2 class="warning">⚠️ WARNING: This action cannot be undone!</h2>
            <p>You are about to reset the entire ImageMatch system. This will:</p>
            <ul>
                <li>Delete <strong>ALL</strong> vectors from the ChromaDB collection</li>
                <li>Delete <strong>ALL</strong> processed images from the server</li>
                <li>Clear the in-memory metadata cache</li>
            </ul>
            
            <p class="warning">All your image data and embeddings will be permanently deleted.</p>
            
            <div class="actions">
                <form action="/reset-system" method="post">
                    <button type="button" class="cancel" onclick="window.location.href='/'">Cancel</button>
                    <button type="submit" class="danger">Yes, Reset Everything</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/reset-system")
async def reset_system_route():
    """Reset the entire system - clear ChromaDB collection and all processed images"""
    logger.info("Reset system request received")
    
    try:
        # Call the reset function
        result = reset_system()
        logger.info("System reset completed successfully")
        
        # Return HTML response
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reset Complete</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .success {{ color: green; font-weight: bold; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                a {{ display: block; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>System Reset Complete</h1>
            
            <div class="section">
                <h2 class="success">✅ System has been reset successfully</h2>
                <p>{result['message']}</p>
                <p>The ImageMatch system has been completely reset. All vectors, metadata, and processed images have been deleted.</p>
                <a href="/">Back to Home</a>
            </div>
        </body>
        </html>
        """)
    except Exception as e:
        logger.error(f"Error during system reset: {e}")
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reset Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .error {{ color: red; font-weight: bold; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                a {{ display: block; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>System Reset Error</h1>
            
            <div class="section">
                <h2 class="error">❌ An error occurred during system reset</h2>
                <p>Error details: {str(e)}</p>
                <a href="/">Back to Home</a>
            </div>
        </body>
        </html>
        """)

@app.get("/edit-metadata/{image_id}")
async def edit_metadata_form(image_id: str):
    """Display form for editing image metadata"""
    logger.info(f"Edit metadata form requested for image: {image_id}")
    
    # Check if image exists
    if image_id not in image_metadata:
        logger.error(f"Image not found: {image_id}")
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .error {{ color: red; font-weight: bold; }}
                a {{ display: block; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1 class="error">Error: Image Not Found</h1>
            <p>The requested image ({image_id}) was not found in the database.</p>
            <a href="/images">Back to Image List</a>
        </body>
        </html>
        """)
    
    # Get metadata for the image
    metadata = image_metadata[image_id]
    
    # Display edit form
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Edit Metadata</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            .image-preview {{ text-align: center; margin-bottom: 20px; }}
            img {{ max-width: 300px; max-height: 300px; }}
            label {{ display: block; margin-top: 15px; font-weight: bold; }}
            input[type="text"] {{ width: 100%; padding: 8px; margin-top: 5px; }}
            textarea {{ width: 100%; height: 150px; padding: 8px; margin-top: 5px; }}
            .ai-caption {{ background-color: #e6f7ff; padding: 15px; border-radius: 5px; border-left: 5px solid #0099ff; margin: 15px 0; }}
            .note {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
            .info-box {{ background-color: #f0f7ff; padding: 10px; border-radius: 5px; border-left: 5px solid #0066cc; margin: 10px 0; }}
            button {{ background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; margin-top: 20px; }}
            button:hover {{ background-color: #45a049; }}
            a {{ display: inline-block; margin: 20px 20px 0 0; }}
        </style>
    </head>
    <body>
        <h1>Edit Image Metadata</h1>
        
        <div class="section">
            <div class="image-preview">
                <img src="/{metadata.get('processed_path', '')}" alt="{metadata.get('description', 'Image')}">
                <p><strong>Image ID:</strong> {image_id}</p>
                <p><strong>Original filename:</strong> {metadata.get('original_filename', 'Unknown')}</p>
            </div>
            
            <div class="info-box">
                <p><strong>Note:</strong> AI-generated captions are automatically included in the custom metadata field for better search results.</p>
            </div>
            
            <form action="/update-metadata/{image_id}" method="post">
                <div>
                    <label for="description">Description:</label>
                    <textarea name="description" id="description" rows="3" required>{metadata['description']}</textarea>
                </div>
                
                <div>
                    <label for="custom_metadata">Custom Metadata:</label>
                    <textarea name="custom_metadata" id="custom_metadata" rows="10">{metadata['custom_metadata'] or ''}</textarea>
                </div>
                
                <p class="note">You can add any additional keywords or information. The AI caption will be automatically included.</p>
                
                <div>
                    <a href="/images">Cancel</a>
                    <button type="submit">Save Changes</button>
                </div>
            </form>
        </div>
    </body>
    </html>
    """)

@app.post("/update-metadata/{image_id}")
async def update_metadata(
    image_id: str, 
    description: str = Form(...), 
    custom_metadata: str = Form(None)
):
    """Update the metadata for an image"""
    logger.info(f"Received request to update metadata for image: {image_id}")
    
    # Verify that the image exists in our system
    if image_id not in image_metadata:
        logger.error(f"Image ID not found: {image_id}")
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # Get current metadata
        current_metadata = image_metadata[image_id]
        
        # Update only the editable fields
        current_metadata['description'] = description
        
        # Update custom metadata (maintaining any AI caption that might be in it)
        current_metadata['custom_metadata'] = custom_metadata if custom_metadata else ""
                
        # Update metadata in ChromaDB too
        logger.info(f"Updating metadata in ChromaDB for image: {image_id}")
        collection.update(
            ids=[image_id],
            metadatas=[current_metadata],
            documents=[description]  # Update document for text search
        )
        logger.info(f"Metadata updated successfully for image: {image_id}")
        
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Metadata Updated</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1 {{ color: #00cc66; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
                p {{ margin: 10px 0; }}
                a {{ display: block; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Metadata Updated Successfully</h1>
                <p>The metadata for image ID: {image_id} has been updated.</p>
                <p>Description: {description}</p>
                <img src="/{current_metadata['processed_path']}" alt="{description}" />
                <a href="/images">Back to Image List</a>
            </div>
        </body>
        </html>
        """)
    
    except Exception as e:
        logger.error(f"Error updating metadata: {e}")
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .error {{ color: red; font-weight: bold; }}
                a {{ display: block; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1 class="error">Error Updating Metadata</h1>
            <p>An error occurred while updating the metadata: {str(e)}</p>
            <a href="/edit-metadata/{image_id}">Back to Edit Form</a>
            <a href="/images">Back to Image List</a>
        </body>
        </html>
        """)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    # Initialize ChromaDB
    logger.info("Application startup: initializing ChromaDB connection")
    try:
        global collection
        collection = init_chromadb()
        # ChromaDB doesn't have a direct parallel to describe_index_stats
        # but we can get some basic info about the collection
        all_ids = collection.get(include=[])["ids"]
        logger.info(f"Successfully connected to ChromaDB collection: {COLLECTION_NAME}")
        logger.info(f"Collection contains {len(all_ids)} vectors")
        
        # Load metadata from ChromaDB
        load_metadata_from_chromadb()
        
    except Exception as e:
        logger.error(f"Warning: Could not initialize ChromaDB: {str(e)}")
        logger.error("Make sure the ChromaDB directory is writable")
    
    # Initialize or verify Moondream
    global moondream_model
    moondream_key = os.getenv("MOONDREAM_API_KEY")
    
    if moondream_key:
        logger.info(f"✅ Moondream API key found and has length {len(moondream_key)}")
        logger.info(f"Moondream API key begins with: {moondream_key[:8]}...")
        
        # Check if moondream_model was successfully initialized
        if not moondream_model:
            logger.info("Moondream model not initialized yet, attempting initialization now...")
            try:
                import moondream as md
                moondream_model = md.vl(api_key=moondream_key)
                logger.info("✅ Moondream model successfully initialized during startup")
            except Exception as e:
                logger.error(f"⚠️ Failed to initialize Moondream model during startup: {e}")
        else:
            logger.info("✅ Moondream model was already initialized")
    else:
        logger.warning("""
        ---------------------------------------------------------------------------------
        MOONDREAM_API_KEY is not set in your .env file. 
        AI image captioning will be disabled.
        
        To enable automatic image captioning:
        1. Get an API key from Moondream Cloud (https://console.moondream.ai/)
        2. Add MOONDREAM_API_KEY=your-api-key to your .env file
        ---------------------------------------------------------------------------------
        """)

# Run the application
if __name__ == "__main__":
    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info") 