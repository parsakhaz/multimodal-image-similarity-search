import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import numpy as np
import time
import json
import imagehash
from datetime import datetime
import uuid
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("image-match-api")

logger.info("Starting ImageMatch API")
logger.info("Importing dependencies...")

# Import AVIF plugin for Pillow
try:
    import pillow_avif
    logging.info("AVIF image support enabled via pillow-avif-plugin")
except ImportError:
    logging.warning("pillow-avif-plugin not found. AVIF image support may be limited.")

# Import utility functions
from .utils import (
    load_clip_model,
    remove_background,
    generate_clip_embedding,
    init_chromadb,
    CLIP_MODEL_ID,
    COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    MAX_TOKEN_LENGTH
)

# Initialize FastAPI
app = FastAPI(title="ImageMatch API")

# Set up CORS for frontend
origins = [
    "http://localhost:3000",  # Next.js default dev server
    "http://127.0.0.1:3000",
    "*",  # Allow all origins for development - restrict this in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Load optional machine learning models depending on environment config
moondream_model = None
try:
    import moondream as md
    from dotenv import load_dotenv
    
    # Load environment variables for API keys
    load_dotenv()
    
    MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")
    if MOONDREAM_API_KEY:
        try:
            # Using API with key
            moondream_model = md.vl(api_key=MOONDREAM_API_KEY)
            logger.info(f"Moondream API initialized successfully with API key: {MOONDREAM_API_KEY[:5]}...")
        except Exception as e:
            logger.error(f"Failed to initialize Moondream API: {e}")
            logger.warning("Will try to initialize local Moondream model instead...")
            try:
                # Fall back to local md.vl without API key
                moondream_model = md.vl()
                logger.info("Initialized local Moondream model for image captioning")
            except Exception as local_e:
                logger.error(f"Failed to initialize local Moondream model: {local_e}")
                logger.warning("Image captioning will be disabled.")
    else:
        # No API key provided, use local model
        try:
            moondream_model = md.vl()
            logger.info("Initialized local Moondream model for image captioning")
        except Exception as e:
            logger.error(f"Failed to initialize local Moondream model: {e}")
            logger.warning("Image captioning will be disabled.")
except ImportError:
    logger.warning("Moondream package not found. Image captioning will be disabled.")

# API routes

@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    description: str = Form(None),
    custom_metadata: str = Form(None),
    remove_bg: bool = Form(False)
):
    """Upload and process an image"""
    logger.info(f"Upload request received: {file.filename}, remove_bg={remove_bg}")
    
    try:
        # Read the uploaded file
        content = await file.read()
        
        # Open the image
        image = Image.open(BytesIO(content))
        
        # Convert to RGB if needed (for RGBA, CMYK or other color modes)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Process the image (embedding generation, etc.)
        metadata, success = process_image(
            image=image,
            filename=file.filename,
            description=description,
            custom_metadata=custom_metadata,
            remove_bg=remove_bg
        )
        
        if success:
            logger.info(f"Upload successful: {metadata['id']}")
            return {"success": True, "metadata": metadata}
        else:
            # This is a duplicate image case, return 409 Conflict with the duplicate metadata
            logger.info(f"Duplicate image detected: {metadata['id']}")
            return JSONResponse(
                status_code=409,
                content={
                    "success": False, 
                    "error": "Duplicate image", 
                    "message": f"This image already exists in the database", 
                    "metadata": metadata
                }
            )
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    filters: List[str] = Form(None)
):
    """Search for similar images using an uploaded image"""
    logger.info(f"Image search request received with file: {file.filename}")
    
    try:
        # Read file content
        content = await file.read()
        
        # Open the image
        image = Image.open(BytesIO(content))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate CLIP embedding for the image
        model, processor = load_clip_model()
        embedding_result = generate_clip_embedding(image=image, model=model, processor=processor)
        
        # Search for similar images
        results = search_similar(
            embedding=embedding_result["image"][0],
            limit=10
        )
        
        # Apply filters if specified
        if filters and len(filters) > 0:
            logger.info(f"Applying filters: {filters}")
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
        
        logger.info(f"Found {len(results)} matches for image search")
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in image search: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/search/text")
async def search_by_text_route(query: str, filters: List[str] = None):
    """Search for images using text query"""
    logger.info(f"Text search request received with query: {query}")
    
    try:
        # Check if we have an empty query but filters are applied
        if not query.strip() and filters and len(filters) > 0:
            logger.info(f"Empty query with filters: {filters}. Returning all images with filters applied.")
            
            # Get all images instead of doing a text search
            results = get_all_images_with_limit(limit=100)  # Higher limit for filter-only searches
        else:
            # Generate CLIP embedding for the text
            model, processor = load_clip_model()
            embedding_result = generate_clip_embedding(text=query, model=model, processor=processor)
            
            # Search for matching images
            results = search_by_text(
                query_text=query,
                limit=10
            )
        
        # Apply filters if specified
        if filters and len(filters) > 0:
            logger.info(f"Applying filters: {filters}")
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
        
        if not query.strip() and filters and len(filters) > 0:
            logger.info(f"Found {len(results)} matches for filter-only search")
        else:
            logger.info(f"Found {len(results)} matches for text search: '{query}'")
            
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in text search: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/search/multimodal")
async def search_multimodal_route(
    file: UploadFile = File(...),
    query: str = Form(...),
    weight_image: float = Form(0.5),
    filters: List[str] = Form(None)
):
    """Search using both image and text"""
    logger.info(f"Multimodal search request received: image={file.filename}, text='{query}', weight={weight_image}")
    
    try:
        # Read file content
        content = await file.read()
        
        # Open the image
        image = Image.open(BytesIO(content))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Perform multimodal search
        results = search_multimodal(
            image=image,
            query_text=query,
            weight_image=weight_image,
            limit=10
        )
        
        # Apply filters if specified
        if filters and len(filters) > 0:
            logger.info(f"Applying filters: {filters}")
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
        
        logger.info(f"Found {len(results)} matches for multimodal search")
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in multimodal search: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/images")
async def get_all_images():
    """Get all images in the database"""
    logger.info("Request for all images received")
    
    try:
        # Extract all metadata
        all_images = list(image_metadata.values())
        
        logger.info(f"Returning {len(all_images)} images")
        return {"images": all_images}
        
    except Exception as e:
        logger.error(f"Error fetching all images: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/filters")
async def get_filters():
    """Get all saved filters"""
    logger.info("Request for filters received")
    
    try:
        filters = load_filters()
        return {"filters": filters}
        
    except Exception as e:
        logger.error(f"Error fetching filters: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/filters")
async def add_filter(
    filter_query: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """Add a new filter"""
    logger.info(f"Add filter request received: {filter_query}")
    
    try:
        # Load existing filters
        filters = load_filters()
        
        # Skip if filter already exists
        if filter_query in filters:
            logger.info(f"Filter already exists: {filter_query}")
            return {"success": True, "message": "Filter already exists", "filters": filters}
        
        # Add new filter
        filters.append(filter_query)
        save_filters(filters)
        
        # Process filter on all images in background
        if background_tasks:
            background_tasks.add_task(process_filter_on_all_images, filter_query)
        
        logger.info(f"Filter added successfully: {filter_query}")
        return {"success": True, "filters": filters}
        
    except Exception as e:
        logger.error(f"Error adding filter: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/api/filters/{filter_query}")
async def delete_filter(filter_query: str):
    """Delete a filter"""
    logger.info(f"Delete filter request received: {filter_query}")
    
    try:
        # Load existing filters
        filters = load_filters()
        
        # Remove filter if it exists
        if filter_query in filters:
            filters.remove(filter_query)
            save_filters(filters)
            logger.info(f"Filter deleted: {filter_query}")
            return {"success": True, "filters": filters}
        else:
            logger.warning(f"Filter not found: {filter_query}")
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Filter not found"}
            )
        
    except Exception as e:
        logger.error(f"Error deleting filter: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/reset")
async def reset_system_route():
    """Reset the system (clear all data)"""
    logger.info("System reset requested")
    
    try:
        success = reset_system()
        
        if success:
            logger.info("System reset successful")
            return {"success": True}
        else:
            logger.error("System reset failed")
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "System reset failed"}
            )
        
    except Exception as e:
        logger.error(f"Error during system reset: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.put("/api/metadata/{image_id}")
async def update_metadata(
    image_id: str, 
    description: str = Form(...), 
    custom_metadata: str = Form(None)
):
    """Update metadata for an image"""
    logger.info(f"Update metadata request received for image: {image_id}")
    
    try:
        # Check if image exists
        if image_id not in image_metadata:
            logger.warning(f"Image not found: {image_id}")
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Image not found"}
            )
        
        # Update metadata
        metadata = image_metadata[image_id]
        metadata["description"] = description
        metadata["custom_metadata"] = custom_metadata
        
        # Save updated metadata
        image_metadata[image_id] = metadata
        
        # Update metadata in ChromaDB
        collection.update(
            ids=[image_id],
            metadatas=[{
                "filename": metadata["filename"],
                "description": description,
                "custom_metadata": custom_metadata
            }]
        )
        
        logger.info(f"Metadata updated for image: {image_id}")
        return {"success": True, "metadata": metadata}
        
    except Exception as e:
        logger.error(f"Error updating metadata: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.on_event("startup")
def startup_event():
    """Initialize on startup"""
    global collection, image_metadata
    
    try:
        # Initialize ChromaDB
        logger.info("Application startup: initializing ChromaDB connection")
        collection = init_chromadb()
        
        # Get basic info about the collection
        all_ids = collection.get(include=[])["ids"]
        logger.info(f"Successfully connected to ChromaDB collection: {COLLECTION_NAME}")
        logger.info(f"Collection contains {len(all_ids)} vectors")
        
        # Load existing metadata from ChromaDB
        image_metadata = load_metadata_from_chromadb()
        
        logger.info(f"Startup complete. Loaded {len(image_metadata)} images from ChromaDB.")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        logger.error("Make sure the ChromaDB directory is writable")
        raise

# Include all the core functions from app.py here
# These are the functions that process images, search, etc.
# They should be moved from app.py without any HTML response functionality

def load_metadata_from_chromadb():
    """Load all image metadata from ChromaDB to initialize our in-memory cache"""
    global collection, image_metadata
    try:
        logger.info("Loading existing metadata from ChromaDB...")
        # Get all IDs stored in the collection
        all_ids = collection.get(include=[])["ids"]
        
        if not all_ids:
            logger.info("No existing metadata found in ChromaDB.")
            return {}
            
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
        return image_metadata
    except Exception as e:
        logger.error(f"Failed to load metadata from ChromaDB: {e}")
        logger.info("Starting with empty metadata cache")
        return {}
    
def generate_image_hash(image: Image.Image) -> str:
    """Generate a perceptual hash of the image to uniquely identify it"""
    # Calculate the perceptual hash of the image
    perceptual_hash = str(imagehash.phash(image))
    return f"img_{perceptual_hash}"
    
def generate_image_caption(image: Image.Image) -> Tuple[Optional[str], Optional[Any]]:
    """Generate a caption for the image using Moondream model if available"""
    global moondream_model
    
    if moondream_model is None:
        logger.info("Moondream model not available, skipping caption generation")
        return None, None
    
    try:
        logger.info("Generating caption with Moondream model")
        start_time = time.time()
        
        # First encode the image
        encoded_input = moondream_model.encode_image(image)
        
        # Use the caption endpoint which is better for captioning
        caption_result = moondream_model.caption(encoded_input)
        caption = caption_result["caption"]
        
        logger.info(f"Caption generated in {time.time() - start_time:.2f} seconds")
        return caption, encoded_input
        
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return None, None
    
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
    global collection, image_metadata, moondream_model
    
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
        description = f"{os.path.splitext(filename)[0]}"
    
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
    
    # Generate URLs for frontend
    # Use absolute URLs with the API server host
    url = f"/static/processed/{image_id}.png"
    thumbnail_url = url
    
    # Store metadata
    metadata = {
        "id": image_id,
        "filename": filename,
        "description": description,
        "custom_metadata": processed_custom_metadata,
        "url": url,
        "thumbnail_url": thumbnail_url,
        "processed_url": processed_path,
        "created_at": datetime.now().isoformat()
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
                    
                    # Use the query method for yes/no questions
                    answer_result = moondream_model.query(encoded_image, formatted_query)
                    answer = answer_result["answer"]
                    
                    logger.info(f"Filter result for {image_id}: {answer}")
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
    
def search_similar(
    embedding: np.ndarray,
    limit: int = 10
) -> List[Dict]:
    """Search for similar images using embedding"""
    global collection, image_metadata
    
    try:
        logger.info(f"Searching for similar images with limit {limit}")
        
        # Query ChromaDB for similar embeddings
        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=limit,
            include=["metadatas", "distances"]
        )
        
        # Process results
        similar_images = []
        
        if not results or "ids" not in results or not results["ids"]:
            logger.info("No similar images found")
            return []
            
        # Extract results
        result_ids = results["ids"][0]
        result_metadatas = results["metadatas"][0]
        result_distances = results["distances"][0]
        
        # Convert distance to similarity score (1 - distance)
        # ChromaDB uses cosine distance, so 0 = identical, 2 = completely different
        # We convert to similarity where 1 = identical, 0 = completely different
        similarities = [1 - (distance / 2) for distance in result_distances]
        
        # Combine results
        for i, img_id in enumerate(result_ids):
            metadata = result_metadatas[i]
            
            # Add similarity score to metadata
            result_metadata = metadata.copy()
            result_metadata["similarity_score"] = similarities[i]
            
            # Update URLs for frontend if needed
            if "url" not in result_metadata:
                result_metadata["url"] = f"/static/processed/{img_id}.png"
            if "thumbnail_url" not in result_metadata:
                result_metadata["thumbnail_url"] = f"/static/processed/{img_id}.png"
            
            similar_images.append(result_metadata)
        
        logger.info(f"Found {len(similar_images)} similar images")
        return similar_images
        
    except Exception as e:
        logger.error(f"Error searching for similar images: {e}")
        return []

def search_by_text(
    query_text: str,
    limit: int = 10
) -> List[Dict]:
    """Search for images using text query"""
    global collection
    
    try:
        logger.info(f"Searching for images matching text: '{query_text}' with limit {limit}")
        
        # Generate text embedding
        model, processor = load_clip_model()
        embedding_result = generate_clip_embedding(text=query_text, model=model, processor=processor)
        text_embedding = embedding_result["text"][0]
        
        # Use the text embedding to find similar images
        return search_similar(embedding=text_embedding, limit=limit)
        
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        return []

def search_multimodal(
    image: Image.Image,
    query_text: str,
    weight_image: float = 0.5,
    limit: int = 10
) -> List[Dict]:
    """Search using both image and text with weighted combination"""
    try:
        logger.info(f"Multimodal search: image weight={weight_image}, text weight={1-weight_image}, limit={limit}")
        
        # Generate embeddings for both image and text
        model, processor = load_clip_model()
        
        # Generate image embedding
        image_embedding_result = generate_clip_embedding(image=image, model=model, processor=processor)
        image_embedding = image_embedding_result["image"][0]
        
        # Generate text embedding
        text_embedding_result = generate_clip_embedding(text=query_text, model=model, processor=processor)
        text_embedding = text_embedding_result["text"][0]
        
        # Combine embeddings with weights
        # First normalize the embeddings to unit length
        image_embedding_norm = image_embedding / np.linalg.norm(image_embedding)
        text_embedding_norm = text_embedding / np.linalg.norm(text_embedding)
        
        # Weighted combination
        combined_embedding = (weight_image * image_embedding_norm + 
                             (1 - weight_image) * text_embedding_norm)
        
        # Normalize the combined embedding
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
        # Search with the combined embedding
        return search_similar(embedding=combined_embedding, limit=limit)
        
    except Exception as e:
        logger.error(f"Error in multimodal search: {e}")
        return []

def load_encoded_image(image_id: str) -> Optional[Any]:
    """Load encoded image from disk if available"""
    import torch
    try:
        encoded_path = f"static/encoded/{image_id}.pt"
        if os.path.exists(encoded_path):
            logger.info(f"Loading encoded image from {encoded_path}")
            # Set weights_only=False for PyTorch 2.6+ compatibility
            # This is safe as we're loading our own generated files
            return torch.load(encoded_path, weights_only=False)
        else:
            logger.info(f"No encoded image found at {encoded_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading encoded image: {e}")
        return None

def load_filters() -> List[str]:
    """Load saved filters from filters.json"""
    try:
        # Use absolute path for filters.json
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filters_file = os.path.join(base_dir, "filters.json")
        
        if os.path.exists(filters_file):
            with open(filters_file, "r") as f:
                filters = json.load(f)
            logger.info(f"Loaded {len(filters)} filters from {filters_file}")
            return filters
        else:
            logger.info(f"No filters file found at {filters_file}, returning empty list")
            return []
    except Exception as e:
        logger.error(f"Error loading filters: {e}")
        return []

def format_filter_query(filter_query: str) -> str:
    """Format filter query to ensure it has yes/no instruction if needed"""
    # If the query already contains "yes" or "no", assume it's already formatted
    lower_query = filter_query.lower()
    if "yes or no:" in lower_query or "yes/no:" in lower_query:
        return filter_query
    
    # Otherwise, add the yes/no instruction
    return f"Yes or No: {filter_query}"

def format_filter_for_display(filter_query: str) -> str:
    """Format filter query for display in the UI"""
    # Remove yes/no instruction if present
    lower_query = filter_query.lower()
    if lower_query.startswith("yes or no:"):
        return filter_query[len("yes or no:"):].strip()
    if lower_query.startswith("yes/no:"):
        return filter_query[len("yes/no:"):].strip()
    
    return filter_query

def save_filters(filters: List[str]) -> None:
    """Save filters to filters.json"""
    try:
        # Use absolute path for filters.json
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filters_file = os.path.join(base_dir, "filters.json")
        
        with open(filters_file, "w") as f:
            json.dump(filters, f)
        logger.info(f"Saved {len(filters)} filters to {filters_file}")
    except Exception as e:
        logger.error(f"Error saving filters: {e}")

def process_filter_on_all_images(filter_query: str) -> None:
    """Process a new filter on all existing images"""
    global collection, image_metadata, moondream_model, filter_progress
    
    try:
        logger.info(f"Processing filter '{filter_query}' on all images")
        
        # Ensure Moondream model is available
        if moondream_model is None:
            logger.error("Moondream model not available, cannot process filter")
            filter_progress[filter_query] = {"status": "error", "message": "Model not available", "progress": 0}
            return
        
        # Format the query for yes/no answer
        formatted_query = format_filter_query(filter_query)
        
        # Get all image IDs
        all_ids = list(image_metadata.keys())
        total_images = len(all_ids)
        logger.info(f"Processing filter on {total_images} images")
        
        # Initialize progress tracking
        results = {}
        
        # Initialize filter progress
        filter_progress[filter_query] = {
            "status": "processing", 
            "progress": 0, 
            "current_image": "",
            "processed": 0,
            "total": total_images
        }
        
        # Process each image
        for idx, image_id in enumerate(all_ids):
            try:
                # Update progress
                progress_percent = int((idx / total_images) * 100) if total_images > 0 else 0
                logger.info(f"Filter progress: {progress_percent}% ({idx}/{total_images})")
                
                # Update progress tracking
                filter_progress[filter_query] = {
                    "status": "processing",
                    "progress": progress_percent,
                    "current_image": image_id,
                    "processed": idx,
                    "total": total_images
                }
                
                # Load encoded image if available
                encoded_input = load_encoded_image(image_id)
                
                if encoded_input is None:
                    # Skip this image if no encoded representation
                    logger.warning(f"No encoded image found for {image_id}, skipping")
                    results[image_id] = "no data"
                    continue
                
                # Apply filter using Moondream
                logger.info(f"Applying filter '{filter_query}' to image {image_id}")
                
                # Use the query method for yes/no questions
                answer_result = moondream_model.query(encoded_input, formatted_query)
                answer = answer_result["answer"]
                    
                logger.info(f"Filter result for {image_id}: {answer}")
                
                # Store result
                results[image_id] = answer.strip() if isinstance(answer, str) else answer
                
                # Update metadata
                metadata = image_metadata.get(image_id, {})
                
                # Initialize or update filter results
                filter_results = {}
                if "filter_results_json" in metadata and metadata["filter_results_json"]:
                    try:
                        filter_results = json.loads(metadata["filter_results_json"])
                    except Exception as e:
                        logger.error(f"Error parsing existing filter results: {e}")
                
                # Add new filter result
                filter_results[filter_query] = results[image_id]
                
                # Save updated filter results
                metadata["filter_results_json"] = json.dumps(filter_results)
                
                # Update in-memory cache
                image_metadata[image_id] = metadata
                
                # Update in ChromaDB
                collection.update(
                    ids=[image_id],
                    metadatas=[metadata]
                )
                
            except Exception as e:
                logger.error(f"Error processing filter for image {image_id}: {e}")
                results[image_id] = "error"
        
        # Update progress to completed
        filter_progress[filter_query] = {
            "status": "completed",
            "progress": 100,
            "processed": total_images,
            "total": total_images
        }
        
        logger.info(f"Filter processing complete for '{filter_query}' on {total_images} images")
        
    except Exception as e:
        logger.error(f"Error processing filter on all images: {e}")
        # Update progress to error state
        filter_progress[filter_query] = {
            "status": "error",
            "message": str(e),
            "progress": 0
        }

def reset_system():
    """Reset the system (clear all data)"""
    global collection, image_metadata
    
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

@app.get("/api/filter-progress")
async def get_filter_progress(filter_query: str):
    """Get progress of a filter being applied to all images"""
    global filter_progress
    
    if filter_query not in filter_progress:
        return {"status": "not_found"}
    
    return filter_progress[filter_query]

@app.post("/api/upload-folder")
async def upload_folder(
    files: List[UploadFile] = File(...),
    remove_bg: bool = Form(False),
    request: Request = None
):
    """Upload multiple images at once"""
    logger.info(f"Folder upload request received with {len(files)} files")
    
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    for file in files:
        try:
            # Read the uploaded file
            content = await file.read()
            
            # Skip empty files
            if not content:
                logger.warning(f"Empty file: {file.filename}, skipping")
                skipped += 1
                results.append({
                    "filename": file.filename,
                    "status": "skipped",
                    "reason": "Empty file"
                })
                continue
            
            # Try to open the image
            try:
                image = Image.open(BytesIO(content))
                
                # Convert to RGB if needed
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
            except Exception as e:
                logger.error(f"Error opening image {file.filename}: {str(e)}")
                failed += 1
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "reason": f"Cannot open image: {str(e)}"
                })
                continue
            
            # Process the image
            metadata, is_new = process_image(
                image=image,
                filename=file.filename,
                remove_bg=remove_bg
            )
            
            if is_new:
                successful += 1
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "id": metadata["id"]
                })
            else:
                skipped += 1
                results.append({
                    "filename": file.filename,
                    "status": "skipped",
                    "reason": "Duplicate image",
                    "id": metadata["id"]
                })
                
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            failed += 1
            results.append({
                "filename": file.filename,
                "status": "error",
                "reason": str(e)
            })
    
    logger.info(f"Folder upload complete: {successful} successful, {skipped} skipped, {failed} failed")
    
    return {
        "success": True,
        "total": len(files),
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
        "results": results
    }

@app.get("/api/image/{image_id}")
async def get_image_by_id(image_id: str):
    """Get image metadata by ID"""
    logger.info(f"Request for image by ID: {image_id}")
    
    try:
        # Check if image exists in our metadata
        if image_id not in image_metadata:
            logger.warning(f"Image not found: {image_id}")
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Image not found"}
            )
        
        # Return the image metadata
        metadata = image_metadata[image_id]
        return {"success": True, "image": metadata}
        
    except Exception as e:
        logger.error(f"Error fetching image by ID: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

def get_all_images_with_limit(limit: int = 100) -> List[Dict]:
    """Get all images with an optional limit"""
    try:
        # Extract all metadata
        all_images = list(image_metadata.values())
        
        # Sort by created_at in descending order (newest first)
        all_images.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Apply limit
        if limit > 0:
            all_images = all_images[:limit]
            
        logger.info(f"Retrieved {len(all_images)} images with limit {limit}")
        return all_images
        
    except Exception as e:
        logger.error(f"Error in get_all_images_with_limit: {e}")
        return [] 