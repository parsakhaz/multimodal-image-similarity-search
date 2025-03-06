# Multimodal Image Similarity Search

An AI-powered image search application that combines text and visual similarity to help you find the right images. This application uses advanced deep learning models (CLIP) to understand both the visual content and textual descriptions of your images.

## New Architecture

This application has been refactored to use a modern architecture:

- **Backend**: FastAPI with ChromaDB for vector storage and similarity search
- **Frontend**: Next.js with React, TypeScript, and Tailwind CSS

## Features

- **Multimodal Search**: Find images using text, visual similarity, or a combination of both
- **Image Management**: Upload, browse, and manage your image collection
- **Custom Metadata**: Add and edit descriptions and custom metadata for each image
- **Background Removal**: Optionally remove image backgrounds during upload
- **Filter System**: Create and apply filters to organize your image collection
- **Automatic Image Captioning**: AI-generated image descriptions for better searchability

## Getting Started

### Prerequisites

- Python 3.8+ for the backend
- Node.js 18+ for the frontend
- Git

### Installation

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/multimodal-image-similarity-search.git
   cd multimodal-image-similarity-search
   ```

2. Set up the backend

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Set up the frontend

   ```bash
   cd frontend
   npm install
   ```

### Configuration

1. Create or edit the `.env` file in the backend directory with the following variables:

   ```txt
   COLLECTION_NAME=image-match
   CHROMA_PERSIST_DIR=chroma_data
   ```

2. Create a `.env.local` file in the frontend directory with:

   ```txt
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

### Running the Application

1. Start the backend (from the backend directory)

   ```bash
   python run.py
   ```

   The API will be available at <http://localhost:8000>

2. Start the frontend development server (from the frontend directory)

   ```bash
   npm run dev
   ```

   The frontend will be available at <http://localhost:3000>

## API Endpoints

The backend provides the following REST API endpoints:

- **POST /api/upload**: Upload and process an image
- **POST /api/search/image**: Search by image similarity
- **GET /api/search/text**: Search by text description
- **POST /api/search/multimodal**: Search using both image and text
- **GET /api/images**: Get all images in the collection
- **GET /api/filters**: Get all saved filters
- **POST /api/filters**: Add a new filter
- **DELETE /api/filters/{filter_query}**: Delete a filter
- **PUT /api/metadata/{image_id}**: Update image metadata
- **POST /api/reset**: Reset the system (clear all data)

## Technologies

### Backend

- FastAPI - High-performance API framework
- ChromaDB - Vector database for similarity search
- PyTorch - Deep learning framework
- CLIP - Multimodal model from OpenAI
- Moondream - Image captioning model
- Rembg - Background removal

### Frontend

- Next.js - React framework
- TypeScript - Type-safe JavaScript
- Tailwind CSS - Utility-first CSS framework
- Zustand - State management
- Axios - HTTP client
- React Dropzone - File upload component

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI's CLIP model](https://github.com/openai/CLIP)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Moondream](https://github.com/vikhyat/moondream)
- [Rembg](https://github.com/danielgatis/rembg)
