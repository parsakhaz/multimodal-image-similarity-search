# ImageMatch: AI-Powered Image and Text Similarity Search with Automatic Captions and Multimodal Capabilities

A powerful image similarity search application that leverages CLIP embeddings and Pinecone for vector storage. This application enables finding similar images based on visual content, text descriptions, or a combination of both.

## Core Features

- **AI-Powered Image Understanding**:
  - CLIP model for semantic image understanding
  - Moondream model for automatic image captioning
  - Background removal with rembg for focused analysis

- **Advanced Search Capabilities**:
  - Image-based similarity search
  - Natural language text queries
  - Multimodal search combining image and text inputs
  - Fast vector similarity search via Pinecone

- **Intelligent Data Management**:
  - Content-based deduplication using perceptual hashing
  - Rich metadata support with custom fields
  - Comprehensive image database interface
  - Stateful architecture with Pinecone integration

- **Technical Features**:
  - Modular, maintainable architecture
  - Detailed logging system
  - Efficient model caching
  - Clean web interface

## Prerequisites

- Python 3.8+
- Pinecone account (free tier supported)
- Sufficient disk space for ML models (~1GB)
- Moondream API key for image captioning

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
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
INDEX_NAME=image-match
MOONDREAM_API_KEY=your_moondream_api_key
```

2. Obtain necessary API keys:
   - Pinecone: Sign up at <https://www.pinecone.io/>
      - Make sure that your image index has a dimension of 768. Other config is default.
   - Moondream: Register at [console.moondream.ai](https://console.moondream.ai/)

### 5. Run the Application

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

2. **Text Search**:
   - Enter natural language descriptions
   - Find images matching semantic concepts
   - Use descriptive phrases for better results

3. **Multimodal Search**:
   - Combine image and text inputs
   - Adjust weights between visual and textual similarity
   - Fine-tune results based on your needs
   - **Note**: Text queries limited to CLIP's 77 token maximum (prioritizes user queries over AI captions)

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
   - CLIP embedding generation (512-dimensional vector)
   - Metadata and embedding storage in Pinecone

2. **Search Processing**:
   - Query processing matches upload pipeline
   - Vector similarity computation via Pinecone
   - Results ranked by similarity scores
   - Optional multimodal query combination

### Performance Optimization

- Model caching reduces loading time
- Background removal improves embedding quality
- Millisecond-level vector search with Pinecone
- Efficient deduplication prevents redundant processing

### Deduplication System

- Perceptual hashing creates content-based fingerprints
- Robust against minor image modifications
- Deterministic content-based identifiers
- Consistent search results for identical images

## Troubleshooting

### Common Issues

1. **Connection Errors**:
   - Verify API keys in `.env`
   - Check internet connectivity
   - Confirm account status

2. **Model Issues**:
   - Ensure sufficient disk space
   - Check memory availability
   - Verify model downloads

3. **Processing Issues**:
   - Convert unsupported formats to JPG/PNG
   - Monitor memory usage for large images
   - Check background removal results

## Current Limitations

### Text Processing Capabilities

- **Extended Token Support**: The system now uses LongCLIP with a 248 token limit (upgraded from CLIP's 77 token limit)
- **Impact on Multimodal Search**: Long text queries and AI captions can now be combined with minimal truncation
- **Solution Implementation**: The system still prioritizes user queries but can now include much more descriptive text

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
