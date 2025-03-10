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
    from fastapi import FastAPI, File, Form, UploadFile, Request, BackgroundTasks, HTTPException, Query, Depends
    from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    from dotenv import load_dotenv
    from fastapi.templating import Jinja2Templates
    
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

# Dictionary to track filter application progress
filter_progress = {}

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
    custom_metadata: Optional[str] = None,
    remove_bg: bool = False
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
    
    # Remove background if requested
    clean_image = image
    if remove_bg:
        try:
            logger.info(f"Removing background for image {image_id}")
            clean_image = remove_background(image)
        except Exception as e:
            logger.error(f"Background removal error: {e}")
            clean_image = image
    else:
        logger.info(f"Skipping background removal for image {image_id}")
    
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
    
    # Apply all existing filters to the new image
    if encoded_image is not None and moondream_model is not None:
        filters = load_filters()
        if filters:
            logger.info(f"Applying {len(filters)} filters to new image {image_id}")
            filter_results = {}
            for filter_query in filters:
                try:
                    # Format the query with yes/no instruction if needed
                    formatted_query = format_filter_query(filter_query)
                    logger.info(f"Applying filter '{filter_query}' to new image")
                    answer = moondream_model.query(encoded_image, formatted_query)["answer"]
                    logger.info(f"Filter result: {answer}")
                    filter_results[filter_query] = answer.strip() if isinstance(answer, str) else answer
                except Exception as e:
                    logger.error(f"Error applying filter '{filter_query}': {e}")
                    filter_results[filter_query] = "error"
            
            # Add filter results to metadata as a JSON string
            if filter_results:
                metadata["filter_results_json"] = json.dumps(filter_results)
                logger.info(f"Added {len(filter_results)} filter results to metadata as JSON")
    
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
            # Explicitly set weights_only=False to allow loading custom classes
            # This is safe since we generated these files ourselves
            return torch.load(encoded_path, weights_only=False)
        else:
            logger.warning(f"Encoded image not found for {image_id}")
            return None
    except Exception as e:
        logger.error(f"Error loading encoded image: {e}")
        return None

def load_filters() -> List[str]:
    """Load the list of dynamic filters from the filters.json file
    
    Returns:
        List of filter queries
    """
    try:
        if os.path.exists("filters.json"):
            logger.info("Loading filters from filters.json")
            with open("filters.json", "r") as f:
                return json.load(f)
        else:
            logger.info("filters.json not found, initializing empty filters list")
            return []
    except Exception as e:
        logger.error(f"Error loading filters: {e}")
        return []

def format_filter_query(filter_query: str) -> str:
    """Format a filter query by appending instructions to answer yes/no if not already present
    
    Args:
        filter_query: The user-provided filter query
        
    Returns:
        Formatted query with "answer either yes or no" instruction appended if needed
    """
    # Check if the query already contains instructions to answer yes/no
    lower_query = filter_query.lower()
    if "answer yes or no" in lower_query or "answer either yes or no" in lower_query:
        return filter_query
    
    # Otherwise, append the instruction
    return f"{filter_query} answer either yes or no"

def format_filter_for_display(filter_query: str) -> str:
    """Format a filter for display in the UI by removing answer instructions
    
    Args:
        filter_query: The filter query to format
        
    Returns:
        Cleaned filter query suitable for UI display
    """
    # Remove the "answer" instructions from the display
    display = filter_query.replace(" answer either yes or no", "")
    display = display.replace(" answer yes or no", "")
    
    # Clean up any trailing punctuation that might look odd after removal
    display = display.rstrip(" ?.")
    
    # Add a question mark if the filter is a question and doesn't end with punctuation
    if any(q in display.lower() for q in ["is ", "are ", "does ", "do ", "has ", "have ", "can ", "could ", "would "]):
        if not display.endswith("?"):
            display += "?"
    
    return display

def save_filters(filters: List[str]) -> None:
    """Save the list of dynamic filters to the filters.json file
    
    Args:
        filters: List of filter queries to save
    """
    try:
        logger.info(f"Saving {len(filters)} filters to filters.json")
        with open("filters.json", "w") as f:
            json.dump(filters, f)
        logger.info("Filters saved successfully")
    except Exception as e:
        logger.error(f"Error saving filters: {e}")

def process_filter_on_all_images(filter_query: str) -> None:
    """Process a single filter query on all existing images
    
    Args:
        filter_query: The filter query to process
    """
    global moondream_model, filter_progress
    
    if not moondream_model:
        logger.warning("Cannot process filter - Moondream not available")
        return
    
    logger.info(f"Processing filter query '{filter_query}' on all images")
    all_ids = collection.get(include=[])["ids"]
    total_images = len(all_ids)
    logger.info(f"Found {total_images} images to process")
    
    # Initialize progress tracking
    filter_progress[filter_query] = {
        "total_count": total_images,
        "processed_count": 0,
        "completed": False
    }
    
    for i, image_id in enumerate(all_ids):
        try:
            # Load the encoded image
            encoded_image = load_encoded_image(image_id)
            if encoded_image is not None:
                # Format the query with yes/no instruction if needed
                formatted_query = format_filter_query(filter_query)
                logger.info(f"Applying filter '{filter_query}' to image {image_id}")
                answer = moondream_model.query(encoded_image, formatted_query)["answer"]
                logger.info(f"Filter result for {image_id}: {answer}")
                
                # Update the metadata
                metadata = collection.get(ids=[image_id], include=["metadatas"])["metadatas"][0]
                
                # Get existing filter results or create empty dict
                filter_results = {}
                if "filter_results_json" in metadata:
                    try:
                        filter_results = json.loads(metadata["filter_results_json"])
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing existing filter_results_json for {image_id}")
                
                # Add new filter result (stripping whitespace for consistent comparison)
                filter_results[filter_query] = answer.strip() if isinstance(answer, str) else answer
                
                # Store back as JSON string
                metadata["filter_results_json"] = json.dumps(filter_results)
                
                # Update ChromaDB
                collection.update(ids=[image_id], metadatas=[metadata])
                
                # Update local cache
                if image_id in image_metadata:
                    image_metadata[image_id] = metadata
            else:
                logger.warning(f"Skipping filter processing for {image_id} - encoded image not found")
        except Exception as e:
            logger.error(f"Error processing filter for image {image_id}: {e}")
        
        # Update progress
        filter_progress[filter_query]["processed_count"] = i + 1
    
    # Mark as completed
    filter_progress[filter_query]["completed"] = True
    logger.info(f"Completed processing filter '{filter_query}' on all images")

# Clear all data (reset function)
def reset_system():
    """Clear all data including stored images and ChromaDB collection"""
    global collection
    global image_metadata
    
    logger.info("Resetting system (clearing all data)")
    
    try:
        # Reset ChromaDB collection
        # Get all IDs in the collection
        all_ids = collection.get(include=[])["ids"]
        
        # Delete all items if there are any
        if all_ids:
            collection.delete(ids=all_ids)
            logger.info("ChromaDB collection has been cleared")
        else:
            logger.info("ChromaDB collection is already empty")
        
        # Reset local cache
        image_metadata = {}
        
        # Delete all processed images
        processed_dir = "static/processed"
        for f in os.listdir(processed_dir):
            if f != ".gitkeep":  # Keep the .gitkeep file
                os.remove(os.path.join(processed_dir, f))
        logger.info("Processed images directory has been cleared")
        
        # Delete all encoded images
        encoded_dir = "static/encoded"
        for f in os.listdir(encoded_dir):
            if f != ".gitkeep":  # Keep the .gitkeep file
                os.remove(os.path.join(encoded_dir, f))
        logger.info("Encoded images directory has been cleared")
        
        # Reset filters.json to empty array
        save_filters([])
        logger.info("Filters have been reset")
        
        return True
    except Exception as e:
        logger.error(f"Error during system reset: {e}")
        return False

# API Routes
@app.get("/", response_class=HTMLResponse)
def home():
    """Simple HTML interface"""
    logger.info("Home page accessed")
    
    # Load existing filters
    filters = load_filters()
    filter_list_html = "<ul>" + "".join([f"<li>{format_filter_for_display(f)}</li>" for f in filters]) + "</ul>" if filters else "<p>No filters defined yet</p>"
    
    # Create filter checkboxes for search forms
    filter_checkboxes = ""
    if filters:
        filter_checkboxes = "<div class='filter-options'><h4>Apply Filters</h4>"
        for f in filters:
            display_text = format_filter_for_display(f)
            filter_checkboxes += f'<label><input type="checkbox" name="filters" value="{f}"> {display_text}</label><br>'
        filter_checkboxes += "</div>"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ImageMatch MVP</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            img {{ max-width: 150px; max-height: 150px; margin: 5px; border: 1px solid #ddd; }}
            .result-item {{ margin-bottom: 20px; }}
            input, textarea, button {{ margin: 10px 0; }}
            textarea {{ width: 100%; height: 100px; }}
            .search-options {{ display: flex; gap: 20px; }}
            .search-box {{ flex: 1; }}
            .highlight {{ background-color: #f0f7ff; border-left: 4px solid #0066cc; padding-left: 15px; }}
            .new-feature {{ background-color: #f0fff0; border-left: 4px solid #00cc66; padding-left: 15px; }}
            .ai-feature {{ background-color: #e6f7ff; border-left: 4px solid #0099ff; padding-left: 15px; }}
            .filter-options {{ margin-top: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
            .button-link {{ display: inline-block; background-color: #4361ee; color: white; padding: 10px 15px; margin: 10px 10px 10px 0; text-decoration: none; border-radius: 5px; }}
            .button-link:hover {{ background-color: #3f37c9; }}
            .app-links {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>ImageMatch</h1>
        
        <div class="new-feature app-links">
            <h3>New UI Pages 🎉</h3>
            <p>Try our improved user interfaces:</p>
            <a href="/app" class="button-link">New Search UI</a>
            <a href="/manage" class="button-link">Manage Filters & Uploads</a>
            <a href="/images" class="button-link">Browse All Images</a>
        </div>
        
        <div class="section highlight" style="background-color: #fff0f5; border-left: 4px solid #ff69b4;">
            <h2>New Dynamic UI Available!</h2>
            <p>Try our new dynamic interface with real-time search and filters - no page reloads required!</p>
            <p><a href="/app" style="display: inline-block; background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">Launch New UI</a></p>
        </div>
        
        <div class="section ai-feature">
            <h2>Automatic Image Captioning</h2>
            <p>We automatically generate descriptive captions for your images using Moondream for better search results.</p>
        </div>
        
        <div class="section new-feature">
            <h2>Dynamic Filters</h2>
            <p>Add custom filters to classify images using AI. Filters will be applied to all existing and future images.</p>
            
            <form action="/add-filter" method="post">
                <input type="text" name="filter_query" placeholder="Enter new filter query (e.g., 'is there red in this image?')" required style="width: 80%;">
                <button type="submit">Add Filter</button>
            </form>
            
            <h3>Existing Filters</h3>
            {filter_list_html}
            <p><small>Note: New filters will be processed in the background on all existing images.</small></p>
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
                        {filter_checkboxes}
                        <button type="submit">Search by Image</button>
                    </form>
                </div>
                
                <div class="search-box">
                    <h3>Search by Text</h3>
                    <p>Enter keywords to find semantically matching images</p>
                    <form action="/search/text" method="get">
                        <input type="text" name="query" placeholder="e.g., red drill, cat, landscape" required><br>
                        {filter_checkboxes}
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
                
                {filter_checkboxes}
                
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
            <!-- Search results will be displayed here -->
        </div>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    description: str = Form(None),
    custom_metadata: str = Form(None),
    remove_bg: bool = Form(False),
    request: Request = None
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
    result, is_new_upload = process_image(image, file.filename, description, custom_metadata, remove_bg)
    logger.info(f"Image processed successfully: {result['id']}")
    
    # Check if request is AJAX (from the manage page)
    is_ajax = request and request.headers.get('accept') == 'application/json'
    
    # Return JSON response for AJAX requests
    if is_ajax:
        return {
            "success": True,
            "is_new_upload": is_new_upload,
            "id": result['id'],
            "description": result['description'],
            "original_path": original_path,
            "processed_path": result['processed_path'],
            "custom_metadata": result['custom_metadata']
        }
    
    # Different status message based on whether the image was new or a duplicate
    status_message = ""
    if is_new_upload:
        status_message = "<h1>Upload Complete</h1>"
    else:
        status_message = '<h1 style="color: orange;">Duplicate Image Detected</h1><p>This image was already in our database. Using existing metadata and embeddings.</p>'
    
    # Remove the separate AI caption section - it's now only in custom metadata
    ai_caption_section = ""
    
    # Return HTML response for regular submissions
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
async def search_by_image(
    file: UploadFile = File(...),
    filters: List[str] = Form(None)
):
    """Search for similar images using an uploaded image"""
    logger.info(f"Image search request received with file: {file.filename}")
    if filters:
        logger.info(f"Filters specified: {filters}")
    
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
    
    # Filter results based on selected filters
    if filters:
        logger.info(f"Filtering results based on {len(filters)} filters")
        filtered_results = []
        for r in results:
            # Get filter results from JSON string
            filter_results = {}
            if "filter_results_json" in r:
                try:
                    filter_results = json.loads(r["filter_results_json"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Error parsing filter_results_json for image {r['id']}")
            
            # Check if this result matches all the selected filters
            if all(filter_results.get(f, "").lower().strip() == "yes" for f in filters):
                filtered_results.append(r)
            else:
                logger.info(f"Image {r['id']} excluded by filter(s)")
        
        # Update results with filtered version
        logger.info(f"Results filtered: {len(results)} -> {len(filtered_results)}")
        results = filtered_results
    
    # Format results for display
    result_html = ""
    if not results:
        result_html = "<p>No matching images found. Try different search criteria or filters.</p>"
    else:
        for r in results:
            similarity_pct = f"{r['similarity'] * 100:.1f}%"
            
            # Add filter results to display if any filters were applied
            filter_display = ""
            if filters and "filter_results_json" in r:
                try:
                    filter_results = json.loads(r["filter_results_json"])
                    filter_display = "<div class='filter-results'><h4>Filter Results:</h4><ul>"
                    for f in filters:
                        answer = filter_results.get(f, "unknown")
                        if isinstance(answer, str):
                            answer = answer.strip()
                        display_text = format_filter_for_display(f)
                        filter_display += f"<li><strong>{display_text}</strong>: {answer}</li>"
                    filter_display += "</ul></div>"
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Error parsing filter_results_json for display, image {r['id']}")
            
            result_html += f"""
            <div class="result">
                <img src="/{r['processed_path']}" alt="{r['description']}">
                <p class="similarity">Similarity: {similarity_pct}</p>
                <p>{r['description']}</p>
                {filter_display}
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
            .filter-results {{ text-align: left; margin-top: 10px; font-size: 0.9em; background-color: #f5f5f5; padding: 5px; }}
            .filter-results ul {{ padding-left: 20px; margin: 5px 0; }}
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
        {f'<div class="applied-filters"><h3>Applied Filters</h3><ul>' + ''.join([f'<li>{f}</li>' for f in filters]) + '</ul></div>' if filters else ''}
        <div id="results">
            {result_html}
        </div>
        <a href="/">Back to Home</a>
    </body>
    </html>
    """)

@app.get("/search/text")
async def search_by_text_route(query: str, filters: List[str] = None):
    """Search for images using text query"""
    logger.info(f"Text search request received with query: '{query}'")
    if filters:
        logger.info(f"Filters specified: {filters}")
    
    # Search using text
    results = search_by_text(query)
    
    # Filter results based on selected filters
    if filters:
        logger.info(f"Filtering results based on {len(filters)} filters")
        filtered_results = []
        for r in results:
            # Get filter results from JSON string
            filter_results = {}
            if "filter_results_json" in r:
                try:
                    filter_results = json.loads(r["filter_results_json"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Error parsing filter_results_json for image {r['id']}")
            
            # Check if this result matches all the selected filters
            if all(filter_results.get(f, "").lower().strip() == "yes" for f in filters):
                filtered_results.append(r)
            else:
                logger.info(f"Image {r['id']} excluded by filter(s)")
        
        # Update results with filtered version
        logger.info(f"Results filtered: {len(results)} -> {len(filtered_results)}")
        results = filtered_results
    
    # Format results for display
    result_html = ""
    if not results:
        result_html = "<p>No matching images found. Try different search criteria or filters.</p>"
    else:
        for r in results:
            similarity_pct = f"{r['similarity'] * 100:.1f}%"
            
            # Add filter results to display if any filters were applied
            filter_display = ""
            if filters and "filter_results_json" in r:
                try:
                    filter_results = json.loads(r["filter_results_json"])
                    filter_display = "<div class='filter-results'><h4>Filter Results:</h4><ul>"
                    for f in filters:
                        answer = filter_results.get(f, "unknown")
                        if isinstance(answer, str):
                            answer = answer.strip()
                        display_text = format_filter_for_display(f)
                        filter_display += f"<li><strong>{display_text}</strong>: {answer}</li>"
                    filter_display += "</ul></div>"
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Error parsing filter_results_json for display, image {r['id']}")
            
            result_html += f"""
            <div class="result">
                <img src="/{r['processed_path']}" alt="{r['description']}">
                <p class="similarity">Similarity: {similarity_pct}</p>
                <p>{r['description']}</p>
                {filter_display}
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
            .filter-results {{ text-align: left; margin-top: 10px; font-size: 0.9em; background-color: #f5f5f5; padding: 5px; }}
            .filter-results ul {{ padding-left: 20px; margin: 5px 0; }}
            a {{ display: block; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Text Search Results</h1>
        <div id="query" style="margin-bottom: 20px;">
            <h3>Query: "{query}"</h3>
        </div>
        {f'<div class="applied-filters"><h3>Applied Filters</h3><ul>' + ''.join([f'<li>{f}</li>' for f in filters]) + '</ul></div>' if filters else ''}
        <div id="results">
            {result_html}
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
                result, is_new_upload = process_image(image, filename, description, custom_metadata, remove_bg=False)
                
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
        custom_metadata = metadata.get('custom_metadata', '')
        
        image_html = f"""
        <div class="card image-item">
            <h3>{metadata.get('description', 'No description')}</h3>
            <div class="image-details">
                <div class="image-preview">
                    <img src="/{metadata.get('processed_path', '')}" alt="{metadata.get('description', 'Image')}">
                </div>
                <div class="image-metadata">
                    <p><strong>ID:</strong> <span class="metadata-value">{image_id}</span></p>
                    <p><strong>Original filename:</strong> <span class="metadata-value">{metadata.get('original_filename', 'Unknown')}</span></p>
                    <p><strong>Upload time:</strong> <span class="metadata-value">{metadata.get('upload_time', 'Unknown')}</span></p>
                    
                    <div class="custom-metadata">
                        <h4>Custom Metadata:</h4>
                        <pre>{custom_metadata}</pre>
                    </div>
                    <div class="actions">
                        <a href="/edit-metadata/{image_id}" class="action-button edit-button">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M11 4H4C3.44772 4 3 4.44772 3 5V19C3 19.5523 3.44772 20 4 20H18C18.5523 20 19 19.5523 19 19V12M17.5858 3.58579C18.3668 2.80474 19.6332 2.80474 20.4142 3.58579C21.1953 4.36683 21.1953 5.63316 20.4142 6.41421L11.8284 15H9L9 12.1716L17.5858 3.58579Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            Edit Metadata
                        </a>
                    </div>
                </div>
            </div>
        </div>
        """
        image_items.append(image_html)
    
    image_list_html = "\n".join(image_items)
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ImageMatch - All Images</title>
        <style>
            :root {{
                --primary-color: #4361ee;
                --secondary-color: #3f37c9;
                --success-color: #4CAF50;
                --text-color: #333;
                --light-bg: #f8f9fa;
                --border-color: #dee2e6;
                --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                --radius: 8px;
            }}
            
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #fff;
                color: var(--text-color);
                line-height: 1.6;
            }}
            
            h1 {{ 
                color: var(--primary-color);
                font-weight: 700;
                margin-bottom: 30px;
                border-bottom: 2px solid var(--border-color);
                padding-bottom: 10px;
            }}
            
            .card {{
                background-color: var(--light-bg);
                padding: 25px;
                border-radius: var(--radius);
                margin-bottom: 30px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border-color);
            }}
            
            .image-item {{
                margin-bottom: 30px;
            }}
            
            .image-item h3 {{
                margin-top: 0;
                color: var(--primary-color);
                font-size: 1.3rem;
                margin-bottom: 15px;
            }}
            
            .image-details {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }}
            
            .image-preview {{
                flex: 0 0 200px;
            }}
            
            .image-metadata {{
                flex: 1;
                min-width: 300px;
            }}
            
            .image-preview img {{
                max-width: 100%;
                border-radius: var(--radius);
                box-shadow: var(--shadow);
            }}
            
            .custom-metadata {{
                background-color: rgba(67, 97, 238, 0.05);
                padding: 15px;
                border-radius: var(--radius);
                margin-top: 15px;
                border-left: 3px solid var(--primary-color);
            }}
            
            .custom-metadata h4 {{
                margin-top: 0;
                color: var(--primary-color);
            }}
            
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
                max-width: 100%;
                overflow-wrap: break-word;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 0.9rem;
                margin: 0;
                padding: 0;
            }}
            
            .metadata-value {{
                font-weight: normal;
                color: #666;
            }}
            
            .actions {{
                margin-top: 20px;
            }}
            
            .action-button {{
                display: inline-flex;
                align-items: center;
                background-color: var(--primary-color);
                color: white;
                padding: 8px 16px;
                text-decoration: none;
                border-radius: var(--radius);
                font-weight: 600;
                transition: background-color 0.3s;
            }}
            
            .action-button svg {{
                margin-right: 8px;
            }}
            
            .action-button:hover {{
                background-color: var(--secondary-color);
                text-decoration: none;
            }}
            
            .edit-button {{
                background-color: var(--success-color);
            }}
            
            .edit-button:hover {{
                background-color: #39873d;
            }}
            
            .nav-bar {{
                display: flex;
                margin-bottom: 30px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 15px;
            }}
            
            .nav-bar a {{
                color: var(--text-color);
                text-decoration: none;
                margin-right: 20px;
                font-weight: 600;
                transition: color 0.2s;
                display: inline-flex;
                align-items: center;
            }}
            
            .nav-bar a:hover {{
                color: var(--primary-color);
            }}
            
            .nav-bar a svg {{
                margin-right: 6px;
            }}
            
            .danger-button {{
                background-color: #dc3545;
                color: white;
                padding: 8px 16px;
                text-decoration: none;
                border-radius: var(--radius);
                font-weight: 600;
                transition: background-color 0.3s;
                display: inline-flex;
                align-items: center;
                margin-top: 20px;
            }}
            
            .danger-button:hover {{
                background-color: #c82333;
                text-decoration: none;
            }}
            
            @media (max-width: 768px) {{
                body {{
                    padding: 15px;
                }}
                
                .card {{
                    padding: 20px;
                }}
                
                .image-details {{
                    flex-direction: column;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>ImageMatch</h1>
        
        <div class="nav-bar">
            <a href="/">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3 12L5 10M5 10L12 3L19 10M5 10V20C5 20.5523 5.44772 21 6 21H9M19 10L21 12M19 10V20C19 20.5523 18.5523 21 18 21H15M9 21C9.55228 21 10 20.5523 10 20V16C10 15.4477 10.4477 15 11 15H13C13.5523 15 14 15.4477 14 16V20C14 20.5523 14.4477 21 15 21M9 21H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Home
            </a>
            <a href="/app">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z" fill="currentColor"/>
                </svg>
                Search
            </a>
            <a href="/manage">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M10.3246 4.31731C10.751 2.5609 13.249 2.5609 13.6754 4.31731C13.9508 5.45193 15.2507 5.99038 16.2478 5.38285C17.7913 4.44239 19.5576 6.2087 18.6172 7.75218C18.0096 8.74925 18.5481 10.0492 19.6827 10.3246C21.4391 10.751 21.4391 13.249 19.6827 13.6754C18.5481 13.9508 18.0096 15.2507 18.6172 16.2478C19.5576 17.7913 17.7913 19.5576 16.2478 18.6172C15.2507 18.0096 13.9508 18.5481 13.6754 19.6827C13.249 21.4391 10.751 21.4391 10.3246 19.6827C10.0492 18.5481 8.74926 18.0096 7.75219 18.6172C6.2087 19.5576 4.44239 17.7913 5.38285 16.2478C5.99038 15.2507 5.45193 13.9508 4.31731 13.6754C2.5609 13.249 2.5609 10.751 4.31731 10.3246C5.45193 10.0492 5.99037 8.74926 5.38285 7.75218C4.44239 6.2087 6.2087 4.44239 7.75219 5.38285C8.74926 5.99037 10.0492 5.45193 10.3246 4.31731Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Manage
            </a>
        </div>
        
        <section>
            <div class="section-header">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <h2>All Stored Images</h2>
            </div>
            
            <div class="card">
                <p>Total images: <strong>{len(image_metadata)}</strong></p>
            </div>
            
            <div class="image-list">
                {image_list_html if image_metadata else '<div class="card"><p>No images found in the database.</p></div>'}
            </div>
        </section>
        
        <a href="/reset-confirm" class="danger-button">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 8px;">
                <path d="M19 7L18.1327 19.1425C18.0579 20.1891 17.187 21 16.1378 21H7.86224C6.81296 21 5.94208 20.1891 5.86732 19.1425L5 7M10 11V17M14 11V17M15 7V4C15 3.44772 14.5523 3 14 3H10C9.44772 3 9 3.44772 9 4V7M4 7H20" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Reset System (Clear All Data)
        </a>
    </body>
    </html>
    """)

@app.get("/reset-confirm", response_class=HTMLResponse)
async def reset_confirm():
    """Show confirmation page before resetting the system"""
    logger.info("Reset confirmation page accessed")
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ImageMatch - Reset Confirmation</title>
        <style>
            :root {
                --primary-color: #4361ee;
                --secondary-color: #3f37c9;
                --success-color: #4CAF50;
                --text-color: #333;
                --light-bg: #f8f9fa;
                --border-color: #dee2e6;
                --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                --radius: 8px;
                --danger-color: #dc3545;
            }
            
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #fff;
                color: var(--text-color);
                line-height: 1.6;
            }
            
            h1 { 
                color: var(--primary-color);
                font-weight: 700;
                margin-bottom: 30px;
                border-bottom: 2px solid var(--border-color);
                padding-bottom: 10px;
            }
            
            .card {
                background-color: var(--light-bg);
                padding: 25px;
                border-radius: var(--radius);
                margin-bottom: 30px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border-color);
            }
            
            .warning { 
                color: var(--danger-color); 
                font-weight: bold; 
            }
            
            .warning-card {
                background-color: rgba(220, 53, 69, 0.05);
                border-left: 3px solid var(--danger-color);
            }
            
            .actions { 
                margin-top: 30px; 
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
            }
            
            button {
                display: inline-flex;
                align-items: center;
                padding: 10px 20px;
                border: none;
                cursor: pointer;
                border-radius: var(--radius);
                font-weight: 600;
                transition: background-color 0.3s;
            }
            
            button svg {
                margin-right: 8px;
            }
            
            button.cancel { 
                background-color: #6c757d; 
                color: white;
            }
            
            button.cancel:hover {
                background-color: #5a6268;
            }
            
            button.danger { 
                background-color: var(--danger-color); 
                color: white; 
            }
            
            button.danger:hover {
                background-color: #c82333;
            }
            
            .nav-bar {
                display: flex;
                margin-bottom: 30px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 15px;
            }
            
            .nav-bar a {
                color: var(--text-color);
                text-decoration: none;
                margin-right: 20px;
                font-weight: 600;
                transition: color 0.2s;
                display: inline-flex;
                align-items: center;
            }
            
            .nav-bar a:hover {
                color: var(--primary-color);
            }
            
            .nav-bar a svg {
                margin-right: 6px;
            }
            
            .section-header {
                display: flex;
                align-items: center;
                margin-bottom: 20px;
            }
            
            .section-header svg {
                margin-right: 10px;
                color: var(--primary-color);
            }
            
            .section-header h2 {
                margin: 0;
                font-size: 1.5rem;
                color: var(--primary-color);
            }
        </style>
    </head>
    <body>
        <h1>ImageMatch</h1>
        
        <div class="nav-bar">
            <a href="/">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3 12L5 10M5 10L12 3L19 10M5 10V20C5 20.5523 5.44772 21 6 21H9M19 10L21 12M19 10V20C19 20.5523 18.5523 21 18 21H15M9 21C9.55228 21 10 20.5523 10 20V16C10 15.4477 10.4477 15 11 15H13C13.5523 15 14 15.4477 14 16V20C14 20.5523 14.4477 21 15 21M9 21H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Home
            </a>
            <a href="/app">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z" fill="currentColor"/>
                </svg>
                Search
            </a>
            <a href="/images">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z" fill="currentColor"/>
                </svg>
                All Images
            </a>
            <a href="/manage">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M10.3246 4.31731C10.751 2.5609 13.249 2.5609 13.6754 4.31731C13.9508 5.45193 15.2507 5.99038 16.2478 5.38285C17.7913 4.44239 19.5576 6.2087 18.6172 7.75218C18.0096 8.74925 18.5481 10.0492 19.6827 10.3246C21.4391 10.751 21.4391 13.249 19.6827 13.6754C18.5481 13.9508 18.0096 15.2507 18.6172 16.2478C19.5576 17.7913 17.7913 19.5576 16.2478 18.6172C15.2507 18.0096 13.9508 18.5481 13.6754 19.6827C13.249 21.4391 10.751 21.4391 10.3246 19.6827C10.0492 18.5481 8.74926 18.0096 7.75219 18.6172C6.2087 19.5576 4.44239 17.7913 5.38285 16.2478C5.99038 15.2507 5.45193 13.9508 4.31731 13.6754C2.5609 13.249 2.5609 10.751 4.31731 10.3246C5.45193 10.0492 5.99037 8.74926 5.38285 7.75218C4.44239 6.2087 6.2087 4.44239 7.75219 5.38285C8.74926 5.99037 10.0492 5.45193 10.3246 4.31731Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Manage
            </a>
        </div>
        
        <section>
            <div class="section-header">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M19 7L18.1327 19.1425C18.0579 20.1891 17.187 21 16.1378 21H7.86224C6.81296 21 5.94208 20.1891 5.86732 19.1425L5 7M10 11V17M14 11V17M15 7V4C15 3.44772 14.5523 3 14 3H10C9.44772 3 9 3.44772 9 4V7M4 7H20" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <h2>System Reset</h2>
            </div>
            
            <div class="card warning-card">
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
                        <button type="button" class="cancel" onclick="window.location.href='/'">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M18 6L6 18M6 6L18 18" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            Cancel
                        </button>
                        <button type="submit" class="danger">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M19 7L18.1327 19.1425C18.0579 20.1891 17.187 21 16.1378 21H7.86224C6.81296 21 5.94208 20.1891 5.86732 19.1425L5 7M10 11V17M14 11V17M15 7V4C15 3.44772 14.5523 3 14 3H10C9.44772 3 9 3.44772 9 4V7M4 7H20" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            Yes, Reset Everything
                        </button>
                    </form>
                </div>
            </div>
        </section>
    </body>
    </html>
    """

@app.post("/reset-system")
async def reset_system_route():
    """Reset the system by clearing all data"""
    logger.info("Reset system request received")
    
    try:
        success = reset_system()
        
        if success:
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>ImageMatch - System Reset Complete</title>
                <meta http-equiv="refresh" content="3;url=/" />
                <style>
                    :root {{
                        --primary-color: #4361ee;
                        --secondary-color: #3f37c9;
                        --success-color: #4CAF50;
                        --text-color: #333;
                        --light-bg: #f8f9fa;
                        --border-color: #dee2e6;
                        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        --radius: 8px;
                    }}
                    
                    body {{ 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 1200px; 
                        margin: 0 auto; 
                        padding: 20px;
                        background-color: #fff;
                        color: var(--text-color);
                        line-height: 1.6;
                    }}
                    
                    h1 {{ 
                        color: var(--primary-color);
                        font-weight: 700;
                        margin-bottom: 30px;
                        border-bottom: 2px solid var(--border-color);
                        padding-bottom: 10px;
                    }}
                    
                    .card {{
                        background-color: var(--light-bg);
                        padding: 25px;
                        border-radius: var(--radius);
                        margin-bottom: 30px;
                        box-shadow: var(--shadow);
                        border: 1px solid var(--border-color);
                    }}
                    
                    .nav-bar {{
                        display: flex;
                        margin-bottom: 30px;
                        border-bottom: 1px solid var(--border-color);
                        padding-bottom: 15px;
                    }}
                    
                    .nav-bar a {{
                        color: var(--text-color);
                        text-decoration: none;
                        margin-right: 20px;
                        font-weight: 600;
                        transition: color 0.2s;
                        display: inline-flex;
                        align-items: center;
                    }}
                    
                    .nav-bar a:hover {{
                        color: var(--primary-color);
                    }}
                    
                    .nav-bar a svg {{
                        margin-right: 6px;
                    }}
                    
                    .section-header {{
                        display: flex;
                        align-items: center;
                        margin-bottom: 20px;
                    }}
                    
                    .section-header svg {{
                        margin-right: 10px;
                        color: var(--primary-color);
                    }}
                    
                    .section-header h2 {{
                        margin: 0;
                        font-size: 1.5rem;
                        color: var(--primary-color);
                    }}
                    
                    .success {{ 
                        color: var(--success-color); 
                    }}
                    
                    .success-card {{
                        background-color: rgba(76, 175, 80, 0.05);
                        border-left: 3px solid var(--success-color);
                    }}
                    
                    .text-center {{
                        text-align: center;
                    }}
                </style>
            </head>
            <body>
                <h1>ImageMatch</h1>
                
                <div class="nav-bar">
                    <a href="/">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M3 12L5 10M5 10L12 3L19 10M5 10V20C5 20.5523 5.44772 21 6 21H9M19 10L21 12M19 10V20C19 20.5523 18.5523 21 18 21H15M9 21C9.55228 21 10 20.5523 10 20V16C10 15.4477 10.4477 15 11 15H13C13.5523 15 14 15.4477 14 16V20C14 20.5523 14.4477 21 15 21M9 21H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        Home
                    </a>
                    <a href="/app">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z" fill="currentColor"/>
                        </svg>
                        Search
                    </a>
                    <a href="/images">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z" fill="currentColor"/>
                        </svg>
                        All Images
                    </a>
                    <a href="/manage">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M10.3246 4.31731C10.751 2.5609 13.249 2.5609 13.6754 4.31731C13.9508 5.45193 15.2507 5.99038 16.2478 5.38285C17.7913 4.44239 19.5576 6.2087 18.6172 7.75218C18.0096 8.74925 18.5481 10.0492 19.6827 10.3246C21.4391 10.751 21.4391 13.249 19.6827 13.6754C18.5481 13.9508 18.0096 15.2507 18.6172 16.2478C19.5576 17.7913 17.7913 19.5576 16.2478 18.6172C15.2507 18.0096 13.9508 18.5481 13.6754 19.6827C13.249 21.4391 10.751 21.4391 10.3246 19.6827C10.0492 18.5481 8.74926 18.0096 7.75219 18.6172C6.2087 19.5576 4.44239 17.7913 5.38285 16.2478C5.99038 15.2507 5.45193 13.9508 4.31731 13.6754C2.5609 13.249 2.5609 10.751 4.31731 10.3246C5.45193 10.0492 5.99037 8.74926 5.38285 7.75218C4.44239 6.2087 6.2087 4.44239 7.75219 5.38285C8.74926 5.99037 10.0492 5.45193 10.3246 4.31731Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        Manage
                    </a>
                </div>
                
                <section>
                    <div class="section-header">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M9 12L11 14L15 10M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        <h2>System Reset Complete</h2>
                    </div>
                    
                    <div class="card success-card text-center">
                        <h3 class="success">✓ All data has been cleared successfully</h3>
                        <p>You will be redirected to the home page in 3 seconds...</p>
                        <p><a href="/">Click here if you are not redirected automatically</a></p>
                    </div>
                </section>
            </body>
            </html>
            """)
        else:
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>ImageMatch - Reset Failed</title>
                <style>
                    :root {{
                        --primary-color: #4361ee;
                        --secondary-color: #3f37c9;
                        --text-color: #333;
                        --light-bg: #f8f9fa;
                        --border-color: #dee2e6;
                        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        --radius: 8px;
                        --danger-color: #dc3545;
                    }}
                    
                    body {{ 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 1200px; 
                        margin: 0 auto; 
                        padding: 20px;
                        background-color: #fff;
                        color: var(--text-color);
                        line-height: 1.6;
                    }}
                    
                    h1 {{ 
                        color: var(--primary-color);
                        font-weight: 700;
                        margin-bottom: 30px;
                        border-bottom: 2px solid var(--border-color);
                        padding-bottom: 10px;
                    }}
                    
                    .card {{
                        background-color: var(--light-bg);
                        padding: 25px;
                        border-radius: var(--radius);
                        margin-bottom: 30px;
                        box-shadow: var(--shadow);
                        border: 1px solid var(--border-color);
                    }}
                    
                    .nav-bar {{
                        display: flex;
                        margin-bottom: 30px;
                        border-bottom: 1px solid var(--border-color);
                        padding-bottom: 15px;
                    }}
                    
                    .nav-bar a {{
                        color: var(--text-color);
                        text-decoration: none;
                        margin-right: 20px;
                        font-weight: 600;
                        transition: color 0.2s;
                        display: inline-flex;
                        align-items: center;
                    }}
                    
                    .nav-bar a:hover {{
                        color: var(--primary-color);
                    }}
                    
                    .nav-bar a svg {{
                        margin-right: 6px;
                    }}
                    
                    .section-header {{
                        display: flex;
                        align-items: center;
                        margin-bottom: 20px;
                    }}
                    
                    .section-header svg {{
                        margin-right: 10px;
                        color: var(--primary-color);
                    }}
                    
                    .section-header h2 {{
                        margin: 0;
                        font-size: 1.5rem;
                        color: var(--primary-color);
                    }}
                    
                    .error {{ 
                        color: var(--danger-color); 
                    }}
                    
                    .error-card {{
                        background-color: rgba(220, 53, 69, 0.05);
                        border-left: 3px solid var(--danger-color);
                    }}
                    
                    .text-center {{
                        text-align: center;
                    }}
                </style>
            </head>
            <body>
                <h1>ImageMatch</h1>
                
                <div class="nav-bar">
                    <a href="/">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M3 12L5 10M5 10L12 3L19 10M5 10V20C5 20.5523 5.44772 21 6 21H9M19 10L21 12M19 10V20C19 20.5523 18.5523 21 18 21H15M9 21C9.55228 21 10 20.5523 10 20V16C10 15.4477 10.4477 15 11 15H13C13.5523 15 14 15.4477 14 16V20C14 20.5523 14.4477 21 15 21M9 21H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        Home
                    </a>
                    <a href="/app">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z" fill="currentColor"/>
                        </svg>
                        Search
                    </a>
                    <a href="/images">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z" fill="currentColor"/>
                        </svg>
                        All Images
                    </a>
                    <a href="/manage">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M10.3246 4.31731C10.751 2.5609 13.249 2.5609 13.6754 4.31731C13.9508 5.45193 15.2507 5.99038 16.2478 5.38285C17.7913 4.44239 19.5576 6.2087 18.6172 7.75218C18.0096 8.74925 18.5481 10.0492 19.6827 10.3246C21.4391 10.751 21.4391 13.249 19.6827 13.6754C18.5481 13.9508 18.0096 15.2507 18.6172 16.2478C19.5576 17.7913 17.7913 19.5576 16.2478 18.6172C15.2507 18.0096 13.9508 18.5481 13.6754 19.6827C13.249 21.4391 10.751 21.4391 10.3246 19.6827C10.0492 18.5481 8.74926 18.0096 7.75219 18.6172C6.2087 19.5576 4.44239 17.7913 5.38285 16.2478C5.99038 15.2507 5.45193 13.9508 4.31731 13.6754C2.5609 13.249 2.5609 10.751 4.31731 10.3246C5.45193 10.0492 5.99037 8.74926 5.38285 7.75218C4.44239 6.2087 6.2087 4.44239 7.75219 5.38285C8.74926 5.99037 10.0492 5.45193 10.3246 4.31731Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        Manage
                    </a>
                </div>
                
                <section>
                    <div class="section-header">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 9V12M12 15H12.01M5.07183 19H18.9282C20.4678 19 21.4301 17.3333 20.6603 16L13.7321 4C12.9623 2.66667 11.0378 2.66667 10.268 4L3.33978 16C2.56998 17.3333 3.53223 19 5.07183 19Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        <h2>System Reset Failed</h2>
                    </div>
                    
                    <div class="card error-card text-center">
                        <h3 class="error">⚠️ There was an error resetting the system</h3>
                        <p>Please try again or contact an administrator if the problem persists.</p>
                        <p><a href="/">Return to Home</a></p>
                    </div>
                </section>
            </body>
            </html>
            """, status_code=500)
    except Exception as e:
        logger.error(f"Error in reset_system_route: {e}")
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ImageMatch - Error</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <h1 class="error">System Error</h1>
            <p>An unexpected error occurred: {str(e)}</p>
            <p><a href="/">Back to Home</a></p>
        </body>
        </html>
        """, status_code=500)

@app.get("/edit-metadata/{image_id}")
async def edit_metadata_form(image_id: str):
    """Display form for editing image metadata"""
    logger.info(f"Edit metadata form requested for image: {image_id}")
    
    # Check if image exists
    if image_id not in image_metadata:
        logger.error(f"Image not found: {image_id}")
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ImageMatch - Image Not Found</title>
            <style>
                :root {{
                    --primary-color: #4361ee;
                    --secondary-color: #3f37c9;
                    --text-color: #333;
                    --light-bg: #f8f9fa;
                    --border-color: #dee2e6;
                    --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    --radius: 8px;
                    --danger-color: #dc3545;
                }}
                
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px; 
                    margin: 0 auto; 
                    padding: 20px;
                    background-color: #fff;
                    color: var(--text-color);
                    line-height: 1.6;
                }}
                
                h1 {{ 
                    color: var(--primary-color);
                    font-weight: 700;
                    margin-bottom: 30px;
                    border-bottom: 2px solid var(--border-color);
                    padding-bottom: 10px;
                }}
                
                .card {{
                    background-color: var(--light-bg);
                    padding: 25px;
                    border-radius: var(--radius);
                    margin-bottom: 30px;
                    box-shadow: var(--shadow);
                    border: 1px solid var(--border-color);
                }}
                
                .nav-bar {{
                    display: flex;
                    margin-bottom: 30px;
                    border-bottom: 1px solid var(--border-color);
                    padding-bottom: 15px;
                }}
                
                .nav-bar a {{
                    color: var(--text-color);
                    text-decoration: none;
                    margin-right: 20px;
                    font-weight: 600;
                    transition: color 0.2s;
                    display: inline-flex;
                    align-items: center;
                }}
                
                .nav-bar a:hover {{
                    color: var(--primary-color);
                }}
                
                .nav-bar a svg {{
                    margin-right: 6px;
                }}
                
                .section-header {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                
                .section-header svg {{
                    margin-right: 10px;
                    color: var(--primary-color);
                }}
                
                .section-header h2 {{
                    margin: 0;
                    font-size: 1.5rem;
                    color: var(--primary-color);
                }}
                
                .error {{ 
                    color: var(--danger-color); 
                }}
                
                .error-card {{
                    background-color: rgba(220, 53, 69, 0.05);
                    border-left: 3px solid var(--danger-color);
                }}
                
                .text-center {{
                    text-align: center;
                }}
                
                a.button {{
                    display: inline-flex;
                    align-items: center;
                    background-color: var(--primary-color);
                    color: white;
                    padding: 10px 20px;
                    border-radius: var(--radius);
                    text-decoration: none;
                    font-weight: 600;
                    margin-top: 20px;
                    transition: background-color 0.3s;
                }}
                
                a.button:hover {{
                    background-color: var(--secondary-color);
                }}
                
                a.button svg {{
                    margin-right: 8px;
                }}
            </style>
        </head>
        <body>
            <h1>ImageMatch</h1>
            
            <div class="nav-bar">
                <a href="/">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 12L5 10M5 10L12 3L19 10M5 10V20C5 20.5523 5.44772 21 6 21H9M19 10L21 12M19 10V20C19 20.5523 18.5523 21 18 21H15M9 21C9.55228 21 10 20.5523 10 20V16C10 15.4477 10.4477 15 11 15H13C13.5523 15 14 15.4477 14 16V20C14 20.5523 14.4477 21 15 21M9 21H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Home
                </a>
                <a href="/app">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z" fill="currentColor"/>
                    </svg>
                    Search
                </a>
                <a href="/images">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z" fill="currentColor"/>
                    </svg>
                    All Images
                </a>
                <a href="/manage">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M10.3246 4.31731C10.751 2.5609 13.249 2.5609 13.6754 4.31731C13.9508 5.45193 15.2507 5.99038 16.2478 5.38285C17.7913 4.44239 19.5576 6.2087 18.6172 7.75218C18.0096 8.74925 18.5481 10.0492 19.6827 10.3246C21.4391 10.751 21.4391 13.249 19.6827 13.6754C18.5481 13.9508 18.0096 15.2507 18.6172 16.2478C19.5576 17.7913 17.7913 19.5576 16.2478 18.6172C15.2507 18.0096 13.9508 18.5481 13.6754 19.6827C13.249 21.4391 10.751 21.4391 10.3246 19.6827C10.0492 18.5481 8.74926 18.0096 7.75219 18.6172C6.2087 19.5576 4.44239 17.7913 5.38285 16.2478C5.99038 15.2507 5.45193 13.9508 4.31731 13.6754C2.5609 13.249 2.5609 10.751 4.31731 10.3246C5.45193 10.0492 5.99037 8.74926 5.38285 7.75218C4.44239 6.2087 6.2087 4.44239 7.75219 5.38285C8.74926 5.99037 10.0492 5.45193 10.3246 4.31731Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Manage
                </a>
            </div>
            
            <section>
                <div class="section-header">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 9V12M12 15H12.01M5.07183 19H18.9282C20.4678 19 21.4301 17.3333 20.6603 16L13.7321 4C12.9623 2.66667 11.0378 2.66667 10.268 4L3.33978 16C2.56998 17.3333 3.53223 19 5.07183 19Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    <h2>Image Not Found</h2>
                </div>
                
                <div class="card error-card text-center">
                    <h3 class="error">⚠️ Image Not Found</h3>
                    <p>The requested image ({image_id}) was not found in the database.</p>
                    
                    <a href="/images" class="button">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z" fill="white"/>
                        </svg>
                        Back to Image List
                    </a>
                </div>
            </section>
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

@app.post("/add-filter")
async def add_filter(
    filter_query: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """Add a new dynamic filter(s) and process them on all existing images
    
    Supports comma-separated filter queries for batch addition
    """
    logger.info(f"Add filter request received: {filter_query}")
    
    if not moondream_model:
        logger.error("Cannot add filter - Moondream model is not available")
        return HTMLResponse(
            content=f"""
            <html>
            <body>
                <h1>Error</h1>
                <p>Cannot add filter - Moondream model is not available.</p>
                <p>Make sure the Moondream API key is correctly configured.</p>
                <a href="/">Back to Home</a>
            </body>
            </html>
            """,
            status_code=400
        )
    
    # Load existing filters
    filters = load_filters()
    
    # Parse comma-separated filter queries
    filter_queries = [query.strip() for query in filter_query.split(',') if query.strip()]
    added_filters = []
    
    for single_query in filter_queries:
        # Check if filter already exists
        if single_query in filters:
            logger.info(f"Filter '{single_query}' already exists, skipping")
            continue
        
        # Add new filter
        filters.append(single_query)
        added_filters.append(single_query)
        logger.info(f"Filter '{single_query}' added successfully")
    
    # Save all filters only once after processing all queries
    if added_filters:
        save_filters(filters)
        
        # Process each new filter on all existing images in the background
        for new_filter in added_filters:
            if background_tasks:
                logger.info(f"Starting background task to process filter '{new_filter}' on all images")
                background_tasks.add_task(process_filter_on_all_images, new_filter)
            else:
                # Process immediately if background tasks are not available
                process_filter_on_all_images(new_filter)
    
    return RedirectResponse(url="/", status_code=303)

@app.post("/delete-filter")
async def delete_filter(filter_query: str = Form(...)):
    """Delete a filter from the filters list
    
    Args:
        filter_query: The filter to delete
    """
    try:
        logger.info(f"Received request to delete filter: {filter_query}")
        # Load current filters
        filters = load_filters()
        
        # Check if filter exists
        if filter_query in filters:
            # Remove the filter
            filters.remove(filter_query)
            save_filters(filters)
            logger.info(f"Filter '{filter_query}' deleted successfully")
            return RedirectResponse(url="/manage", status_code=303)
        else:
            logger.warning(f"Filter '{filter_query}' not found in filters list")
            return HTMLResponse(
                """
                <html>
                <head>
                    <title>Error</title>
                    <meta http-equiv="refresh" content="3;url=/manage">
                    <style>
                        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                        .error { color: #d32f2f; }
                    </style>
                </head>
                <body>
                    <h1 class="error">Error</h1>
                    <p>Filter not found. Redirecting back...</p>
                    <a href="/manage">Return to Manage page</a>
                </body>
                </html>
                """,
                status_code=404
            )
    except Exception as e:
        logger.error(f"Error deleting filter: {e}")
        return HTMLResponse(
            """
            <html>
            <head>
                <title>Error</title>
                <meta http-equiv="refresh" content="3;url=/manage">
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                    .error { color: #d32f2f; }
                </style>
            </head>
            <body>
                <h1 class="error">Error</h1>
                <p>An error occurred while deleting the filter. Redirecting back...</p>
                <a href="/manage">Return to Manage page</a>
            </body>
            </html>
            """,
            status_code=500
        )

@app.post("/search/multimodal")
async def search_by_multimodal(
    file: UploadFile = File(...),
    query: str = Form(...),
    weight_image: float = Form(0.5),
    filters: List[str] = Form(None)
):
    """Search for images using both uploaded image and text query"""
    logger.info(f"Multimodal search request received with file: {file.filename}, text: '{query}'")
    if filters:
        logger.info(f"Filters specified: {filters}")
    
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
    
    # Filter results based on selected filters
    if filters:
        logger.info(f"Filtering results based on {len(filters)} filters")
        filtered_results = []
        for r in results:
            # Get filter results from JSON string
            filter_results = {}
            if "filter_results_json" in r:
                try:
                    filter_results = json.loads(r["filter_results_json"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Error parsing filter_results_json for image {r['id']}")
            
            # Check if this result matches all the selected filters
            if all(filter_results.get(f, "").lower().strip() == "yes" for f in filters):
                filtered_results.append(r)
            else:
                logger.info(f"Image {r['id']} excluded by filter(s)")
        
        # Update results with filtered version
        logger.info(f"Results filtered: {len(results)} -> {len(filtered_results)}")
        results = filtered_results
    
    # Format results for display
    result_html = ""
    if not results:
        result_html = "<p>No matching images found. Try different search criteria or filters.</p>"
    else:
        for r in results:
            similarity_pct = f"{r['similarity'] * 100:.1f}%"
            
            # Add filter results to display if any filters were applied
            filter_display = ""
            if filters and "filter_results_json" in r:
                try:
                    filter_results = json.loads(r["filter_results_json"])
                    filter_display = "<div class='filter-results'><h4>Filter Results</h4><ul class='filter-results-list'>"
                    for f in filters:
                        answer = filter_results.get(f, "unknown")
                        if isinstance(answer, str):
                            answer = answer.strip()
                        display_text = format_filter_for_display(f)
                        
                        # Replace yes/no with SVG icons
                        icon_html = ""
                        if answer.lower() == "yes":
                            icon_html = """<svg class="filter-icon filter-yes" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M5 13L9 17L19 7" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>"""
                        elif answer.lower() == "no":
                            icon_html = """<svg class="filter-icon filter-no" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M6 18L18 6M6 6L18 18" stroke="#F44336" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>"""
                        else:
                            icon_html = """<svg class="filter-icon filter-unknown" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 16V16.01M12 13C12.5523 13 13 12.5523 13 12C13 11.4477 12.5523 11 12 11C11.4477 11 11 11.4477 11 12C11 12.5523 11.4477 13 12 13ZM12 13V8M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="#FFC107" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>"""
                        
                        filter_display += f"<li><strong>{display_text}</strong> {icon_html}</li>"
                    filter_display += "</ul></div>"
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Error parsing filter_results_json for display, image {r['id']}")
            
            result_html += f"""
            <div class="card result">
                <div class="image-details">
                    <div class="image-preview">
                        <img src="/{r['processed_path']}" alt="{r['description']}">
                        <p class="similarity">Similarity: {similarity_pct}</p>
                    </div>
                    <div class="image-metadata">
                        <h3>{r['description']}</h3>
                        <p><strong>ID:</strong> <span class="metadata-value">{r['id']}</span></p>
                        {filter_display}
                    </div>
                </div>
            </div>
            """
    
    # Combine query image and results with styling
    final_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ImageMatch - Search Results</title>
        <style>
            :root {{
                --primary-color: #4361ee;
                --secondary-color: #3f37c9;
                --success-color: #4CAF50;
                --error-color: #F44336;
                --warning-color: #FFC107;
                --text-color: #333;
                --light-bg: #f8f9fa;
                --border-color: #dee2e6;
                --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                --radius: 8px;
            }}
            
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #fff;
                color: var(--text-color);
                line-height: 1.6;
            }}
            
            h1 {{ 
                color: var(--primary-color);
                font-weight: 700;
                margin-bottom: 30px;
                border-bottom: 2px solid var(--border-color);
                padding-bottom: 10px;
            }}

            h3 {{
                margin-top: 0;
                color: var(--primary-color);
            }}
            
            .card {{
                background-color: var(--light-bg);
                padding: 25px;
                border-radius: var(--radius);
                margin-bottom: 30px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border-color);
                height: 100%;
                display: flex;
                flex-direction: column;
            }}
            
            .search-results {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 25px;
            }}
            
            @media (min-width: 768px) {{
                .search-results {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}
            
            @media (min-width: 1024px) {{
                .search-results {{
                    grid-template-columns: repeat(3, 1fr);
                }}
            }}
            
            .image-details {{
                display: flex;
                flex-direction: column;
                height: 100%;
            }}
            
            .image-preview {{
                margin-bottom: 15px;
            }}
            
            .image-metadata {{
                flex: 1;
            }}
            
            .image-preview img {{
                width: 100%;
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                object-fit: cover;
                aspect-ratio: 4/3;
            }}
            
            .query-image {{
                margin-bottom: 20px;
            }}
            
            .query-image img {{
                max-width: 200px;
                border-radius: var(--radius);
                box-shadow: var(--shadow);
            }}
            
            .similarity {{
                margin-top: 10px;
                font-weight: bold;
                color: var(--primary-color);
            }}
            
            .applied-filters {{
                background-color: rgba(67, 97, 238, 0.05);
                padding: 15px;
                border-radius: var(--radius);
                margin-bottom: 20px;
                border-left: 3px solid var(--primary-color);
            }}
            
            .applied-filters h3 {{
                margin-top: 0;
                color: var(--primary-color);
            }}
            
            .applied-filters ul {{
                margin: 0;
                padding-left: 20px;
            }}
            
            .filter-results {{
                background-color: rgba(67, 97, 238, 0.05);
                padding: 15px;
                border-radius: var(--radius);
                margin-top: 15px;
            }}
            
            .filter-results h4 {{
                margin-top: 0;
                color: var(--primary-color);
            }}
            
            .filter-results-list {{
                margin: 0;
                padding-left: 0;
                list-style-type: none;
            }}
            
            .filter-results-list li {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 5px;
                padding-bottom: 5px;
                border-bottom: 1px solid rgba(0,0,0,0.05);
            }}
            
            .filter-results-list li:last-child {{
                margin-bottom: 0;
                padding-bottom: 0;
                border-bottom: none;
            }}
            
            .filter-icon {{
                margin-left: 10px;
            }}
            
            .filter-yes {{
                color: var(--success-color);
            }}
            
            .filter-no {{
                color: var(--error-color);
            }}
            
            .filter-unknown {{
                color: var(--warning-color);
            }}
            
            .metadata-value {{
                font-weight: normal;
                color: #666;
            }}
            
            .nav-bar {{
                display: flex;
                margin-bottom: 30px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 15px;
            }}
            
            .nav-bar a {{
                color: var(--text-color);
                text-decoration: none;
                margin-right: 20px;
                font-weight: 600;
                transition: color 0.2s;
                display: inline-flex;
                align-items: center;
            }}
            
            .nav-bar a:hover {{
                color: var(--primary-color);
            }}
            
            .nav-bar a svg {{
                margin-right: 6px;
            }}
            
            @media (max-width: 768px) {{
                body {{
                    padding: 15px;
                }}
                
                .card {{
                    padding: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>ImageMatch</h1>
        
        <div class="nav-bar">
            <a href="/">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3 12L5 10M5 10L12 3L19 10M5 10V20C5 20.5523 5.44772 21 6 21H9M19 10L21 12M19 10V20C19 20.5523 18.5523 21 18 21H15M9 21C9.55228 21 10 20.5523 10 20V16C10 15.4477 10.4477 15 11 15H13C13.5523 15 14 15.4477 14 16V20C14 20.5523 14.4477 21 15 21M9 21H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Home
            </a>
            <a href="/app">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z" fill="currentColor"/>
                </svg>
                Search
            </a>
            <a href="/images">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z" fill="currentColor"/>
                </svg>
                All Images
            </a>
            <a href="/manage">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M10.3246 4.31731C10.751 2.5609 13.249 2.5609 13.6754 4.31731C13.9508 5.45193 15.2507 5.99038 16.2478 5.38285C17.7913 4.44239 19.5576 6.2087 18.6172 7.75218C18.0096 8.74925 18.5481 10.0492 19.6827 10.3246C21.4391 10.751 21.4391 13.249 19.6827 13.6754C18.5481 13.9508 18.0096 15.2507 18.6172 16.2478C19.5576 17.7913 17.7913 19.5576 16.2478 18.6172C15.2507 18.0096 13.9508 18.5481 13.6754 19.6827C13.249 21.4391 10.751 21.4391 10.3246 19.6827C10.0492 18.5481 8.74926 18.0096 7.75219 18.6172C6.2087 19.5576 4.44239 17.7913 5.38285 16.2478C5.99038 15.2507 5.45193 13.9508 4.31731 13.6754C2.5609 13.249 2.5609 10.751 4.31731 10.3246C5.45193 10.0492 5.99037 8.74926 5.38285 7.75218C4.44239 6.2087 6.2087 4.44239 7.75219 5.38285C8.74926 5.99037 10.0492 5.45193 10.3246 4.31731Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Manage
            </a>
        </div>
        
        <section>
            <div class="section-header">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <h2>Search Results</h2>
            </div>

            {f'<div class="applied-filters"><h3>Applied Filters</h3><ul>' + ''.join([f'<li>{f}</li>' for f in filters]) + '</ul></div>' if filters else ''}
            
            <div class="search-results">
                {result_html}
            </div>
            
            <a href="/images" class="action-button" style="display: inline-flex; align-items: center; background-color: var(--primary-color); color: white; padding: 8px 16px; text-decoration: none; border-radius: 8px; margin-top: 20px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 8px;">
                    <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Back to All Images
            </a>
        </section>
    </body>
    </html>
    """
    
    return HTMLResponse(final_html)

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

# Add this near other global variables after app initialization
templates = Jinja2Templates(directory="templates")

@app.get("/app", response_class=HTMLResponse)
async def main_app(request: Request):
    """Serve the new dynamic UI using htmx"""
    filters = load_filters()
    return templates.TemplateResponse("main.html", {"request": request, "filters": filters})

@app.get("/manage", response_class=HTMLResponse)
async def manage_app(request: Request):
    """Serve the page for managing filters and uploads"""
    filters = load_filters()
    return templates.TemplateResponse("manage.html", {"request": request, "filters": filters})

@app.post("/search")
async def unified_search(
    file: UploadFile = File(None),
    query: str = Form(None),
    weight_image: float = Form(0.5),
    filters: List[str] = Form(None)
):
    """Unified search endpoint that handles all search types (image, text, multimodal)
    and returns partial HTML for htmx to update the UI."""
    file_name = file.filename if file else None
    logger.info(f"Unified search request received. Query: '{query}', File: {file_name}")
    if filters:
        logger.info(f"Filters specified: {filters}")
    
    filters = filters or []
    results = []
    query_image_html = ""
    
    # Check if file is valid and has content
    has_valid_file = False
    file_content = None
    
    if file is not None:
        try:
            file_content = await file.read()
            # Only consider the file valid if it has content
            has_valid_file = len(file_content) > 0
            # Reset file cursor for further reads if needed
            if has_valid_file:
                await file.seek(0)
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            has_valid_file = False
    
    # Determine the type of search based on provided parameters
    if has_valid_file and not query:
        # Image-only search
        try:
            image = Image.open(BytesIO(file_content))
            
            # Handle image mode conversion if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Remove background (optional for search)
            try:
                clean_image = remove_background(image)
            except Exception as e:
                logger.warning(f"Background removal failed, using original image: {str(e)}")
                clean_image = image
            
            # Generate embedding and search
            embeddings = generate_clip_embedding(clean_image)
            results = search_similar(embeddings["image"][0])
            
            # Create HTML for the query image display
            query_image_html = f"""
            <div class="query-image">
                <h3>Query Image</h3>
                <img src="data:image/jpeg;base64,{base64.b64encode(file_content).decode()}">
            </div>
            """
        except Exception as e:
            logger.error(f"Error processing image file: {str(e)}")
            return HTMLResponse(f"<p>Error processing image: {str(e)}</p>")
        
    elif query and not has_valid_file:
        # Text-only search
        results = search_by_text(query)
        
    elif has_valid_file and query:
        # Multimodal search (combine image and text)
        try:
            image = Image.open(BytesIO(file_content))
            
            # Handle image mode conversion if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Weight validation
            weight_image = min(max(weight_image, 0.0), 1.0)  # Clamp between 0 and 1
            
            # Perform multimodal search
            results = search_multimodal(image, query, weight_image)
            
            # Create HTML for the query image display
            query_image_html = f"""
            <div class="query-image">
                <h3>Query Image</h3>
                <img src="data:image/jpeg;base64,{base64.b64encode(file_content).decode()}">
                <p>Text Query: "{query}" (Image Weight: {weight_image})</p>
            </div>
            """
        except Exception as e:
            logger.error(f"Error processing image file for multimodal search: {str(e)}")
            return HTMLResponse(f"<p>Error processing image for multimodal search: {str(e)}</p>")
    else:
        # No valid search parameters
        if not query and not has_valid_file:
            return HTMLResponse("<p>Please provide an image, text, or both to search.</p>")
    
    # Filter results based on selected filters
    if filters:
        logger.info(f"Filtering results based on {len(filters)} filters")
        filtered_results = []
        for r in results:
            # Get filter results from JSON string
            filter_results = {}
            if "filter_results_json" in r:
                try:
                    filter_results = json.loads(r["filter_results_json"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Error parsing filter_results_json for image {r['id']}")
            
            # Check if this result matches all the selected filters
            if all(filter_results.get(f, "").lower().strip() == "yes" for f in filters):
                filtered_results.append(r)
        
        # Update results with filtered version
        logger.info(f"Results filtered: {len(results)} -> {len(filtered_results)}")
        results = filtered_results
    
    # Format results for display
    result_html = ""
    if not results:
        result_html = "<p>No matching images found. Try different search criteria or filters.</p>"
    else:
        for r in results:
            similarity_pct = f"{r['similarity'] * 100:.1f}%"
            
            # Add filter results to display if any filters were applied
            filter_display = ""
            if filters and "filter_results_json" in r:
                try:
                    filter_results = json.loads(r["filter_results_json"])
                    filter_display = "<div class='filter-results'><h4>Filter Results</h4><ul class='filter-results-list'>"
                    for f in filters:
                        answer = filter_results.get(f, "unknown")
                        if isinstance(answer, str):
                            answer = answer.strip()
                        display_text = format_filter_for_display(f)
                        
                        # Replace yes/no with SVG icons
                        icon_html = ""
                        if answer.lower() == "yes":
                            icon_html = """<svg class="filter-icon filter-yes" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M5 13L9 17L19 7" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>"""
                        elif answer.lower() == "no":
                            icon_html = """<svg class="filter-icon filter-no" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M6 18L18 6M6 6L18 18" stroke="#F44336" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>"""
                        else:
                            icon_html = """<svg class="filter-icon filter-unknown" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 16V16.01M12 13C12.5523 13 13 12.5523 13 12C13 11.4477 12.5523 11 12 11C11.4477 11 11 11.4477 11 12C11 12.5523 11.4477 13 12 13ZM12 13V8M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="#FFC107" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>"""
                        
                        filter_display += f"<li><strong>{display_text}</strong> {icon_html}</li>"
                    filter_display += "</ul></div>"
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Error parsing filter_results_json for display, image {r['id']}")
            
            result_html += f"""
            <div class="card result">
                <div class="image-details">
                    <div class="image-preview">
                        <img src="/{r['processed_path']}" alt="{r['description']}">
                        <p class="similarity">Similarity: {similarity_pct}</p>
                    </div>
                    <div class="image-metadata">
                        <h3>{r['description']}</h3>
                        <p><strong>ID:</strong> <span class="metadata-value">{r['id']}</span></p>
                        {filter_display}
                    </div>
                </div>
            </div>
            """
    
    # Combine query image and results with styling
    final_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ImageMatch - Search Results</title>
        <style>
            :root {{
                --primary-color: #4361ee;
                --secondary-color: #3f37c9;
                --success-color: #4CAF50;
                --error-color: #F44336;
                --warning-color: #FFC107;
                --text-color: #333;
                --light-bg: #f8f9fa;
                --border-color: #dee2e6;
                --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                --radius: 8px;
            }}
            
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #fff;
                color: var(--text-color);
                line-height: 1.6;
            }}
            
            h1 {{ 
                color: var(--primary-color);
                font-weight: 700;
                margin-bottom: 30px;
                border-bottom: 2px solid var(--border-color);
                padding-bottom: 10px;
            }}

            h3 {{
                margin-top: 0;
                color: var(--primary-color);
            }}
            
            .card {{
                background-color: var(--light-bg);
                padding: 25px;
                border-radius: var(--radius);
                margin-bottom: 30px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border-color);
                height: 100%;
                display: flex;
                flex-direction: column;
            }}
            
            .search-results {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 25px;
            }}
            
            @media (min-width: 768px) {{
                .search-results {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}
            
            @media (min-width: 1024px) {{
                .search-results {{
                    grid-template-columns: repeat(3, 1fr);
                }}
            }}
            
            .image-details {{
                display: flex;
                flex-direction: column;
                height: 100%;
            }}
            
            .image-preview {{
                margin-bottom: 15px;
            }}
            
            .image-metadata {{
                flex: 1;
            }}
            
            .image-preview img {{
                width: 100%;
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                object-fit: cover;
                aspect-ratio: 4/3;
            }}
            
            .query-image {{
                margin-bottom: 20px;
            }}
            
            .query-image img {{
                max-width: 200px;
                border-radius: var(--radius);
                box-shadow: var(--shadow);
            }}
            
            .similarity {{
                margin-top: 10px;
                font-weight: bold;
                color: var(--primary-color);
            }}
            
            .applied-filters {{
                background-color: rgba(67, 97, 238, 0.05);
                padding: 15px;
                border-radius: var(--radius);
                margin-bottom: 20px;
                border-left: 3px solid var(--primary-color);
            }}
            
            .applied-filters h3 {{
                margin-top: 0;
                color: var(--primary-color);
            }}
            
            .applied-filters ul {{
                margin: 0;
                padding-left: 20px;
            }}
            
            .filter-results {{
                background-color: rgba(67, 97, 238, 0.05);
                padding: 15px;
                border-radius: var(--radius);
                margin-top: 15px;
            }}
            
            .filter-results h4 {{
                margin-top: 0;
                color: var(--primary-color);
            }}
            
            .filter-results-list {{
                margin: 0;
                padding-left: 0;
                list-style-type: none;
            }}
            
            .filter-results-list li {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 5px;
                padding-bottom: 5px;
                border-bottom: 1px solid rgba(0,0,0,0.05);
            }}
            
            .filter-results-list li:last-child {{
                margin-bottom: 0;
                padding-bottom: 0;
                border-bottom: none;
            }}
            
            .filter-icon {{
                margin-left: 10px;
            }}
            
            .filter-yes {{
                color: var(--success-color);
            }}
            
            .filter-no {{
                color: var(--error-color);
            }}
            
            .filter-unknown {{
                color: var(--warning-color);
            }}
            
            .metadata-value {{
                font-weight: normal;
                color: #666;
            }}
            
            .nav-bar {{
                display: flex;
                margin-bottom: 30px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 15px;
            }}
            
            .nav-bar a {{
                color: var(--text-color);
                text-decoration: none;
                margin-right: 20px;
                font-weight: 600;
                transition: color 0.2s;
                display: inline-flex;
                align-items: center;
            }}
            
            .nav-bar a:hover {{
                color: var(--primary-color);
            }}
            
            .nav-bar a svg {{
                margin-right: 6px;
            }}
            
            @media (max-width: 768px) {{
                body {{
                    padding: 15px;
                }}
                
                .card {{
                    padding: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>ImageMatch</h1>
        
        <div class="nav-bar">
            <a href="/">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3 12L5 10M5 10L12 3L19 10M5 10V20C5 20.5523 5.44772 21 6 21H9M19 10L21 12M19 10V20C19 20.5523 18.5523 21 18 21H15M9 21C9.55228 21 10 20.5523 10 20V16C10 15.4477 10.4477 15 11 15H13C13.5523 15 14 15.4477 14 16V20C14 20.5523 14.4477 21 15 21M9 21H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Home
            </a>
            <a href="/app">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z" fill="currentColor"/>
                </svg>
                Search
            </a>
            <a href="/images">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z" fill="currentColor"/>
                </svg>
                All Images
            </a>
            <a href="/manage">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M10.3246 4.31731C10.751 2.5609 13.249 2.5609 13.6754 4.31731C13.9508 5.45193 15.2507 5.99038 16.2478 5.38285C17.7913 4.44239 19.5576 6.2087 18.6172 7.75218C18.0096 8.74925 18.5481 10.0492 19.6827 10.3246C21.4391 10.751 21.4391 13.249 19.6827 13.6754C18.5481 13.9508 18.0096 15.2507 18.6172 16.2478C19.5576 17.7913 17.7913 19.5576 16.2478 18.6172C15.2507 18.0096 13.9508 18.5481 13.6754 19.6827C13.249 21.4391 10.751 21.4391 10.3246 19.6827C10.0492 18.5481 8.74926 18.0096 7.75219 18.6172C6.2087 19.5576 4.44239 17.7913 5.38285 16.2478C5.99038 15.2507 5.45193 13.9508 4.31731 13.6754C2.5609 13.249 2.5609 10.751 4.31731 10.3246C5.45193 10.0492 5.99037 8.74926 5.38285 7.75218C4.44239 6.2087 6.2087 4.44239 7.75219 5.38285C8.74926 5.99037 10.0492 5.45193 10.3246 4.31731Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Manage
            </a>
        </div>
        
        <section>
            <div class="section-header">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <h2>Search Results</h2>
            </div>

            {query_image_html}
            {f'<div class="applied-filters"><h3>Applied Filters</h3><ul>' + ''.join([f'<li>{f}</li>' for f in filters]) + '</ul></div>' if filters else ''}
            
            <div class="search-results">
                {result_html}
            </div>
            
            <a href="/images" class="action-button" style="display: inline-flex; align-items: center; background-color: var(--primary-color); color: white; padding: 8px 16px; text-decoration: none; border-radius: 8px; margin-top: 20px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 8px;">
                    <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Back to All Images
            </a>
        </section>
    </body>
    </html>
    """
    
    return HTMLResponse(final_html)

@app.get("/filter-progress")
async def get_filter_progress(filter_query: str):
    """Get the progress of a filter being applied to all images
    
    Args:
        filter_query: The filter query to check progress for
        
    Returns:
        Progress information including total count, processed count, and completion status
    """
    global filter_progress
    
    if filter_query not in filter_progress:
        return JSONResponse({
            "total_count": 0,
            "processed_count": 0,
            "completed": True
        })
    
    return JSONResponse(filter_progress[filter_query])

@app.post("/upload-folder")
async def upload_folder(
    files: List[UploadFile] = File(...),
    remove_bg: bool = Form(False),
    request: Request = None
):
    """Upload and process multiple images from a folder"""
    logger.info(f"Folder upload request received with {len(files)} files")
    
    results = []
    processed_count = 0
    failed_count = 0
    
    for file in files:
        try:
            # Skip non-image files
            if not file.content_type or 'image' not in file.content_type:
                logger.warning(f"Skipping non-image file: {file.filename} with content type {file.content_type}")
                failed_count += 1
                continue
                
            logger.info(f"Processing file {processed_count + 1}/{len(files)}: {file.filename}")
            
            # Extract just the base filename, ignoring any subdirectories
            base_filename = os.path.basename(file.filename)
            
            # Read file content
            content = await file.read()
            
            # Handle image opening with error handling
            try:
                # Try opening the image directly
                image = Image.open(BytesIO(content))
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
            except Exception as e:
                logger.error(f"Error opening image {file.filename}: {str(e)}")
                failed_count += 1
                continue
            
            # Save original image using just the base filename
            original_path = f"static/uploads/{base_filename}"
            with open(original_path, "wb") as f:
                f.write(content)
            
            # Process image - using AI-generated description since this is a batch upload
            # We don't pass any manual description or metadata
            result, is_new_upload = process_image(image, base_filename, None, None, remove_bg)
            
            results.append({
                "filename": file.filename,
                "id": result['id'],
                "is_new_upload": is_new_upload,
                "description": result.get('description', ''),
                "custom_metadata": result.get('custom_metadata', '')
            })
            
            processed_count += 1
            
            # Update progress in a real system, you might use WebSockets for this
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            failed_count += 1
            
    return {
        "success": True,
        "processed": processed_count,
        "failed": failed_count,
        "results": results
    }

# Run the application
if __name__ == "__main__":
    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info") 