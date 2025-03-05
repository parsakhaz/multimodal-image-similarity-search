# ImageMatch: AI-Powered Image and Text Similarity Search with Automatic Captions and Multimodal Capabilities

A powerful image similarity search application that leverages CLIP embeddings and ChromaDB for vector storage. This application enables finding similar images based on visual content, text descriptions, or a combination of both

## Core Features

- **AI-Powered Image Understanding**:
  - CLIP model for semantic image understanding
  - Moondream model for automatic image captioning
  - Background removal with rembg for focused analysis

- **Advanced Search Capabilities**:
  - Image-based similarity search
  - Natural language text queries
  - Multimodal search combining image and text inputs
  - Dynamic filters for attribute-based filtering
  - Fast vector similarity search via ChromaDB

- **Intelligent Data Management**:
  - Content-based deduplication using perceptual hashing
  - Rich metadata support with custom fields
  - Comprehensive image database interface
  - Stateful architecture with local ChromaDB integration

- **Technical Features**:
  - Modular, maintainable architecture
  - Detailed logging system
  - Efficient model caching
  - Clean, consistent web interface with standardized navigation
  - Unified design language across all application pages

## Prerequisites

- Python 3.8+
- Sufficient disk space for ML models (~1GB)
- ChromaDB v0.6.0+ for vector storage
- Moondream API key for image captioning (optional)

## Setup Instructions

### 1. Clone Repository

```bash
git clone <repository-url>
cd image-match
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

1. Create a `.env` file with your credentials:

```env
COLLECTION_NAME=image-match
CHROMA_PERSIST_DIR=chroma_data
MOONDREAM_API_KEY=your_moondream_api_key
```

2. Obtain necessary API keys:
   - Moondream: Register at [console.moondream.ai](https://console.moondream.ai/) (optional for image captioning)

### 5. Initialize Database

```bash
python init_db.py
```

This script creates the ChromaDB collection and verifies that everything is set up correctly.

### 6. Run the Application

```bash
python app.py
```

Access the application at <http://localhost:8000>

## Usage Guide

### Image Management

1. **Upload Images**:
   - Use the upload form on the homepage
   - Add descriptions and custom metadata
   - Automatic duplicate detection prevents redundant storage
   - AI-generated captions enhance searchability

2. **Batch Upload**:
   - Visit `/upload-samples` to process multiple images
   - Automatically handles images from the `images` directory
   - Provides detailed upload status reports

3. **Database Management**:
   - Browse all stored images at `/images`
   - View and edit image metadata
   - Monitor content-based hash IDs
   - Track AI-generated captions

### Search Functionality

1. **Image Search**:
   - Upload a query image
   - View results ranked by similarity
   - Examine similarity scores
   - Apply dynamic filters to refine results

2. **Text Search**:
   - Enter natural language descriptions
   - Find images matching semantic concepts
   - Use descriptive phrases for better results
   - Combine with dynamic filters for precise results

3. **Multimodal Search**:
   - Combine image and text inputs
   - Adjust weights between visual and textual similarity
   - Fine-tune results based on your needs
   - Apply dynamic filters to narrow down results
   - **Note**: Uses LongCLIP with extended token support (248 tokens)

### System Management

1. **Reset Functionality**:
   - Clear all stored data when needed
   - Confirmation process prevents accidents
   - Fresh start capability for testing

## Technical Details

### Image Processing Pipeline

1. **Upload Processing**:
   - Perceptual hash calculation for deduplication
   - Background removal for focused analysis
   - CLIP embedding generation (768-dimensional vector)
   - Metadata and embedding storage in ChromaDB

2. **Search Processing**:
   - Query processing matches upload pipeline
   - Vector similarity computation via ChromaDB
   - Results ranked by similarity scores
   - Optional multimodal query combination

### Performance Optimization

- Model caching reduces loading time
- Background removal improves embedding quality
- Efficient local vector storage with ChromaDB
- Efficient deduplication prevents redundant processing
- Encoded image storage reduces computational overhead for AI captioning

### Deduplication System

- Perceptual hashing creates content-based fingerprints
- Robust against minor image modifications
- Deterministic content-based identifiers
- Consistent search results for identical images

### ChromaDB Integration

- Local vector database with persistent storage
- No need for external API keys or cloud services
- Compatible with ChromaDB v0.6.0+
- Data stored in the `chroma_data` directory by default

### Encoded Image Optimization

- Moondream-encoded images are saved during initial processing
- Reduces computational overhead for repetitive AI operations
- Encoded tensors stored as `.pt` files in `static/encoded` directory
- Automatically managed alongside other image data
- Significantly improves performance for multimodal operations

### Dynamic Filters System

- Define custom queries that can be applied to all images
- Queries are processed by Moondream model to generate yes/no answers
- Results are stored as metadata and used during search operations
- Filter results displayed alongside search results for transparency
- Filters automatically applied to newly uploaded images
- Ability to process filters on all existing images
- Progress modal with real-time tracking when applying filters
- Visual feedback prevents interruption during filter processing
- JSON-based storage of filter results for efficient querying

### Utility Scripts

- `init_db.py`: Initialize ChromaDB collection and verify setup

## Troubleshooting

### Common Issues

1. **Storage Errors**:
   - Verify `chroma_data` directory is writable
   - Check disk space availability
   - Ensure proper permissions

2. **Model Issues**:
   - Ensure sufficient disk space
   - Check memory availability
   - Verify model downloads

3. **Processing Issues**:
   - Convert unsupported formats to JPG/PNG
   - Monitor memory usage for large images
   - Check background removal results

4. **ChromaDB Issues**:
   - Verify ChromaDB v0.6.0+ is installed
   - Check for ChromaDB compatibility errors in logs

## Current Limitations

### Text Processing Capabilities

- **Extended Token Support**: The system uses LongCLIP with a 248 token limit (upgraded from CLIP's 77 token limit)
- **Impact on Multimodal Search**: Long text queries and AI captions can now be combined with minimal truncation
- **Solution Implementation**: The system prioritizes user queries but can now include much more descriptive text

### Other Considerations

- Background removal may occasionally remove important visual elements
- Very large image collections may experience performance degradation
- Some uncommon image formats may require conversion before upload

## Development Roadmap

- Hybrid search combining vector and text capabilities
- Advanced metadata filtering and validation
- User authentication system
- Batch processing optimization
- REST API development
- Enhanced preprocessing options
- **Semantic Image Segmentation**: Support for region-based search within images
- **Fine-tuned Embeddings**: Domain-specific model adaptation for specialized collections
