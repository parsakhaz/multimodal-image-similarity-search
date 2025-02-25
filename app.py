# ImageMatch MVP: A Simple Image Similarity Search Tool
# Uses CLIP embeddings and Pinecone for storage and search

import os
import uuid
import base64
import logging
import time
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
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
        init_pinecone,
        CLIP_MODEL_ID,
        PINECONE_API_KEY,
        PINECONE_CLOUD,
        PINECONE_REGION,
        INDEX_NAME
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
app.mount("/static", StaticFiles(directory="static"), name="static")
logger.info("Directories setup complete")

logger.info(f"Using Pinecone config: Cloud={PINECONE_CLOUD}, Region={PINECONE_REGION}, Index={INDEX_NAME}")

# Image metadata storage (in-memory for simplicity)
# In a production app, you might want to use a proper database for this
image_metadata = {}

# Load existing metadata from Pinecone
def load_metadata_from_pinecone():
    """Load all image metadata from Pinecone to initialize our in-memory cache"""
    try:
        logger.info("Loading existing metadata from Pinecone...")
        index = init_pinecone()
        # Note: This is not efficient for large collections as it fetches everything
        # For production, you would implement pagination or a proper database
        query_response = index.query(
            vector=[0] * 512,  # Dummy vector for metadata-only query
            top_k=10000,  # Adjust based on your expected collection size
            include_metadata=True
        )
        
        # Extract metadata - handle both old and new API formats
        try:
            # Try new API format
            if hasattr(query_response, 'matches'):
                for match in query_response.matches:
                    image_id = match.id
                    metadata = match.metadata
                    image_metadata[image_id] = metadata
            else:
                # Fall back to old format
                for match in query_response['matches']:
                    image_id = match['id']
                    metadata = match['metadata']
                    image_metadata[image_id] = metadata
        except (AttributeError, TypeError):
            logger.warning("Could not process query response from Pinecone - format may have changed")
            
        logger.info(f"Loaded metadata for {len(image_metadata)} existing images")
    except Exception as e:
        logger.error(f"Failed to load metadata from Pinecone: {e}")
        logger.info("Starting with empty metadata cache")

# Load existing metadata on startup
load_metadata_from_pinecone()

# Generate a deterministic ID for the image based on its content
def generate_image_hash(image: Image.Image) -> str:
    """Generate a perceptual hash of the image to uniquely identify it"""
    # Calculate the perceptual hash of the image
    phash = str(imagehash.phash(image))
    logger.info(f"Generated perceptual hash for image: {phash}")
    return phash

# Generate a caption for an image using Moondream
def generate_image_caption(image: Image.Image) -> str:
    """Generate a caption for the image using Moondream API"""
    if moondream_model is None:
        logger.warning("Moondream model not available. Using generic caption.")
        return None
    
    try:
        logger.info("Generating image caption with Moondream")
        # Encode the image
        encoded_image = moondream_model.encode_image(image)
        
        # Generate the caption
        result = moondream_model.caption(encoded_image)
        caption = result["caption"]
        
        logger.info(f"Generated caption: {caption}")
        return caption
    except Exception as e:
        logger.error(f"Error generating image caption: {e}")
        return None

# Process and store image
def process_image(
    image: Image.Image,
    filename: str,
    description: Optional[str] = None,
    custom_metadata: Optional[str] = None
) -> Tuple[Dict, bool]:
    """Process image and store in Pinecone
    
    Returns:
        Tuple containing (metadata, is_new_upload)
    """
    # Generate content-based ID
    image_id = generate_image_hash(image)
    
    # Check if this image already exists in our system
    index = init_pinecone()
    existing_check = index.fetch([image_id])
    
    # If the image already exists, return its metadata
    # Handle both old dictionary format and new FetchResponse object
    if existing_check:
        try:
            # New Pinecone API (FetchResponse object)
            if hasattr(existing_check, 'vectors') and image_id in existing_check.vectors:
                logger.info(f"Image with hash {image_id} already exists, skipping processing")
                return existing_check.vectors[image_id].metadata, False
        except AttributeError:
            # Fall back to old dictionary format
            if existing_check.get('vectors', {}).get(image_id):
                logger.info(f"Image with hash {image_id} already exists, skipping processing")
                return existing_check['vectors'][image_id]['metadata'], False
    
    # Generate caption for the image using Moondream
    generated_caption = generate_image_caption(image)
    
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
    
    # Use generated caption if available and no description provided
    if not description and generated_caption:
        description = generated_caption
    elif not description:
        # Fall back to filename as simple description if no caption was generated
        description = f"An image of {os.path.splitext(filename)[0]}"
    
    # Prepare custom metadata, appending AI caption if available
    processed_custom_metadata = custom_metadata or ""
    if generated_caption:
        # Add a separator if there's existing custom metadata
        if processed_custom_metadata:
            processed_custom_metadata += "\n\n"
        # Add just the caption text without the "AI Caption:" prefix
        processed_custom_metadata += generated_caption
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
        "ai_caption": generated_caption or "No caption generated",
        "custom_metadata": processed_custom_metadata,
        "upload_time": datetime.now().isoformat()
    }
    
    # Store in Pinecone
    logger.info(f"Storing image {image_id} in Pinecone index")
    index.upsert(
        vectors=[
            {
                "id": image_id,
                "values": clip_embedding,
                "metadata": metadata
            }
        ]
    )
    logger.info(f"Image {image_id} stored successfully")
    
    # Store metadata locally for quick access
    image_metadata[image_id] = metadata
    
    return metadata, True

# Search for similar images
def search_similar(
    embedding: np.ndarray,
    limit: int = 10
) -> List[Dict]:
    """Search for similar images by vector similarity using Pinecone"""
    logger.info(f"Searching for similar images (limit: {limit})")
    index = init_pinecone()
    
    # Query the index
    start_time = time.time()
    results = index.query(
        vector=embedding.tolist(),
        top_k=limit,
        include_metadata=True
    )
    logger.info(f"Search completed in {time.time() - start_time:.2f} seconds")
    
    # Format results - handle both old and new API formats
    formatted_results = []
    try:
        # Try new API format
        if hasattr(results, 'matches'):
            for match in results.matches:
                result = match.metadata
                result["similarity"] = match.score
                formatted_results.append(result)
        else:
            # Fall back to old format
            for match in results["matches"]:
                result = match["metadata"]
                result["similarity"] = match["score"]
                formatted_results.append(result)
    except (AttributeError, TypeError) as e:
        logger.error(f"Error processing search results: {e}")
    
    logger.info(f"Found {len(formatted_results)} similar images")
    return formatted_results

# Text search for images
def search_by_text(
    query_text: str,
    limit: int = 10
) -> List[Dict]:
    """Search for images using text query"""
    logger.info(f"Text search with query: '{query_text}'")
    
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

# Clear all data (reset function)
def reset_system():
    """Reset the entire system by clearing Pinecone index and processed images"""
    logger.info("Resetting system - clearing all data")
    
    # Clear Pinecone index
    try:
        index = init_pinecone()
        
        # Get all vector IDs
        query_response = index.query(
            vector=[0] * 512,  # Dummy vector for metadata-only query
            top_k=10000,  # Adjust based on your expected collection size
            include_metadata=True
        )
        
        # Extract all IDs to delete - handle both old and new API formats
        ids_to_delete = []
        try:
            # Try new API format
            if hasattr(query_response, 'matches'):
                ids_to_delete = [match.id for match in query_response.matches]
            else:
                # Fall back to old format
                ids_to_delete = [match['id'] for match in query_response['matches']]
        except (AttributeError, TypeError):
            logger.warning("Could not process query response from Pinecone - format may have changed")
        
        if ids_to_delete:
            # Delete vectors in batches (Pinecone has limits on batch size)
            batch_size = 1000
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i+batch_size]
                index.delete(ids=batch)
            
            logger.info(f"Deleted {len(ids_to_delete)} vectors from Pinecone index")
        else:
            logger.info("No vectors found in Pinecone index to delete")
            
    except Exception as e:
        logger.error(f"Error clearing Pinecone index: {e}")
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
    except Exception as e:
        logger.error(f"Error clearing processed images: {e}")
        raise
    
    # Clear in-memory metadata cache
    global image_metadata
    image_metadata = {}
    logger.info("Cleared in-memory metadata cache")
    
    return {"success": True, "message": f"System reset complete. Deleted {len(ids_to_delete) if ids_to_delete else 0} vectors and {count if 'count' in locals() else 0} image files."}

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
    
    # Display AI caption section if available
    ai_caption_section = ""
    if result.get('ai_caption') and result['ai_caption'] != "No caption generated":
        ai_caption_section = f"""
        <div class="ai-caption">
            <h3>AI-Generated Caption:</h3>
            <p style="font-style: italic;">"{result['ai_caption']}"</p>
        </div>
        """
    
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
        
        {ai_caption_section}
        
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
    
    # Search using both image and text
    results = search_multimodal(image, query, weight_image=weight_image)
    
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
                <p>"{query}"</p>
                
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
    
    # If metadata cache is empty, try to reload from Pinecone
    if not image_metadata:
        load_metadata_from_pinecone()
    
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
                <li>Delete <strong>ALL</strong> vectors from the Pinecone index</li>
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
    """Reset the entire system - clear Pinecone index and all processed images"""
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
    
    # Check if AI caption is in custom metadata already
    ai_caption = metadata.get('ai_caption', '')
    custom_metadata = metadata.get('custom_metadata', '')
    
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
                <label for="description">Description:</label>
                <input type="text" id="description" name="description" value="{metadata.get('description', '')}" required>
                
                <div class="ai-caption">
                    <label for="ai_caption">AI-Generated Caption:</label>
                    <input type="text" id="ai_caption" name="ai_caption" value="{metadata.get('ai_caption', 'No caption generated')}">
                    <p class="note">This caption will be displayed separately and included in the custom metadata field.</p>
                </div>
                
                <label for="custom_metadata">Custom Metadata:</label>
                <textarea id="custom_metadata" name="custom_metadata">{metadata.get('custom_metadata', '')}</textarea>
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
    ai_caption: str = Form(None),
    custom_metadata: str = Form(None)
):
    """Update the metadata for an image"""
    logger.info(f"Updating metadata for image: {image_id}")
    
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
    
    try:
        # Get current metadata
        current_metadata = image_metadata[image_id]
        
        # Update only the editable fields
        current_metadata['description'] = description
        
        # Update AI caption if provided
        if ai_caption is not None:
            current_metadata['ai_caption'] = ai_caption
        
        # Process custom metadata, ensuring AI caption is included
        processed_custom_metadata = custom_metadata if custom_metadata else ""
        
        # Check if AI caption exists and should be included in custom metadata
        ai_caption_to_use = current_metadata.get('ai_caption', None)
        if ai_caption_to_use and ai_caption_to_use != "No caption generated":
            # Don't add AI caption if it's already in the custom metadata
            if ai_caption_to_use not in processed_custom_metadata:
                # Add a separator if there's existing custom metadata
                if processed_custom_metadata:
                    processed_custom_metadata += "\n\n"
                processed_custom_metadata += ai_caption_to_use
                logger.info(f"Added AI caption to custom metadata during update: {ai_caption_to_use}")
        
        current_metadata['custom_metadata'] = processed_custom_metadata
        
        # Update Pinecone
        index = init_pinecone()
        
        # Get current vector data
        fetch_response = index.fetch([image_id])
        vector_data = None
        
        # Handle both old and new API formats
        try:
            # Try new API format
            if hasattr(fetch_response, 'vectors'):
                vector_data = fetch_response.vectors[image_id].values
            # Fall back to old format
            else:
                vector_data = fetch_response['vectors'][image_id]['values']
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"Error retrieving vector data: {e}")
            raise Exception("Could not retrieve vector data for updating metadata")
        
        # Update vector with new metadata
        index.upsert(
            vectors=[
                {
                    "id": image_id,
                    "values": vector_data,
                    "metadata": current_metadata
                }
            ]
        )
        
        # Update in-memory cache
        image_metadata[image_id] = current_metadata
        
        logger.info(f"Metadata updated successfully for image: {image_id}")
        
        # Return success response
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Metadata Updated</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .success {{ color: green; font-weight: bold; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                img {{ max-width: 300px; max-height: 300px; }}
                a {{ display: block; margin: 20px 0; }}
                .ai-caption {{ background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1 class="success">Metadata Updated Successfully</h1>
            
            <div class="section">
                <p>The metadata for image <strong>{image_id}</strong> has been updated successfully.</p>
                <p><strong>Description:</strong> {description}</p>
                
                <div class="ai-caption">
                    <p><strong>AI Caption:</strong> {current_metadata.get('ai_caption', 'No caption')}</p>
                </div>
                
                <div>
                    <a href="/images">Back to Image List</a>
                    <a href="/edit-metadata/{image_id}">Edit Again</a>
                </div>
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
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
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
    # Initialize Pinecone
    logger.info("Application startup: initializing Pinecone connection")
    try:
        index = init_pinecone()
        stats = index.describe_index_stats()
        logger.info(f"Successfully connected to Pinecone index: {INDEX_NAME}")
        logger.info(f"Index stats: {stats}")
    except Exception as e:
        logger.error(f"Warning: Could not initialize Pinecone: {str(e)}")
        logger.error("You'll need to set PINECONE_API_KEY in .env file")
    
    # Verify Moondream API key
    moondream_key = os.getenv("MOONDREAM_API_KEY")
    if moondream_key:
        logger.info(f"✅ Moondream API key found and has length {len(moondream_key)}")
        logger.info(f"Moondream API key begins with: {moondream_key[:8]}...")
        # Check if moondream_model was successfully initialized
        if not moondream_model:
            logger.warning("⚠️ Moondream API key found but model was not initialized. Check earlier logs for errors.")
    else:
        logger.warning("""
        ---------------------------------------------------------------------------------
        MOONDREAM_API_KEY is not set in your .env file. 
        AI image captioning will be disabled.
        
        To enable automatic image captioning:
        1. Get an API key from Moondream Cloud (https://moonshot.ai/)
        2. Add MOONDREAM_API_KEY=your-api-key to your .env file
        ---------------------------------------------------------------------------------
        """)

# Run the application
if __name__ == "__main__":
    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info") 