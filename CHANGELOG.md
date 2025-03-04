# Changelog

All notable changes to the ImageMatch project will be documented in this file.

## [2.1.0] - 2024-03-05

Performance optimization with encoded image storage.

### Added
- Persistent storage for Moondream-encoded images to reduce computational overhead
- New directory `static/encoded` for storing encoded image tensors
- Helper function for loading encoded images for reuse in AI operations

### Changed
- Modified image captioning process to save encoded representations
- Enhanced reset functionality to manage encoded image files
- Optimized `generate_image_caption` function to return both caption and encoded image

### Technical
- Implemented PyTorch tensor storage for encoded images
- Integrated encoded image handling into the image processing pipeline
- Ensured proper cleanup of encoded images during system reset

## [2.0.0] - 2024-03-04

Major architecture change: Migrated from Pinecone to ChromaDB for vector database.

### Added
- Local ChromaDB integration replacing cloud-based Pinecone
- New `init_db.py` script for initializing the ChromaDB collection
- Detailed logging for ChromaDB operations

### Changed
- Replaced all Pinecone API calls with ChromaDB equivalents
- Updated metadata handling to ensure AI captions are consistently stored
- Modified startup process to properly initialize ChromaDB collection
- Enhanced AI caption integration with metadata for better search results
- Updated `process_image` function to properly store AI captions in metadata
- Improved Moondream model initialization in application startup
- Made application compatible with ChromaDB v0.6.0 API changes

### Removed
- Pinecone dependency and all Pinecone-specific code
- Cloud-based vector storage in favor of local persistence

### Technical
- Added ChromaDB as core dependency (v0.6.0+)
- Implemented persistent vector storage in local `chroma_data/` directory
- Updated environment variable handling for ChromaDB configuration
- Fixed API compatibility issues with ChromaDB's collection management

## [1.5.0] - 2024-03-04

Major enhancement to text processing capabilities with LongCLIP.

### Added
- Implemented LongCLIP model with extended token support (248 tokens vs. 77)
- Significantly increased text query capacity for more comprehensive descriptions
- Better handling of combined queries with AI captions without truncation

### Changed
- Upgraded from standard CLIP model to LongCLIP for improved performance
- Enhanced multimodal search to leverage longer text contexts
- Improved token limit handling throughout the application
- Updated logging to provide better visibility into token usage

### Fixed
- Removed unnecessary truncation of captions in multimodal search
- Eliminated ellipsis characters that could affect embedding quality

## [1.4.1] - 2024-03-03

Enhanced search transparency.

### Added
- Display of the exact final query used for CLIP embedding generation in multimodal search results
- Improved visibility of how user queries and AI captions are combined for search

### Changed
- Enhanced multimodal search results page with a dedicated section showing the actual text used for embedding

## [1.4.0] - 2024-03-02

Improved multimodal search query handling.

### Changed
- Significantly improved how text queries are handled in multimodal search
- Now prioritizing user's explicit query text over AI-generated captions
- Enhanced token allocation logic to ensure user queries are preserved in full whenever possible
- Added transparent UI feedback showing whether caption was included in search or not

### Fixed
- Further refined the solution to token limit issues in multimodal search
- Better handling of longer user queries by dynamically adjusting caption inclusion

## [1.3.9] - 2024-03-01

Fixed token limit issue in multimodal search.

### Fixed
- Resolved error when combined text query exceeded CLIP's token limit (77 tokens)
- Implemented intelligent text truncation for both user queries and AI captions
- Added safety mechanism to ensure queries remain within model constraints
- Improved handling of caption display to show full caption while using truncated version for search

## [1.3.8] - 2024-02-29

Enhanced multimodal search functionality with automatic image captioning.

### Added
- Automatic caption generation for images used in multimodal search
- Integration of AI-generated captions into text queries for more comprehensive search results
- Visual display of generated captions in search results for transparency

### Changed
- Improved multimodal search to leverage both user-provided text and AI-generated descriptions
- Enhanced search results presentation to show original query and AI captions separately

## [1.3.7] - 2024-02-28

Enhanced metadata handling and display for better search results.

### Changed
- Removed "AI Caption:" prefix from the custom metadata field to improve text embeddings
- Captions are now added directly to the custom metadata field without prefix
- This enhancement improves the semantic search results when using text queries

### Fixed
- Fixed word wrap issues in custom metadata display
- Added improved CSS styling for metadata display throughout the application
- Enhanced text overflow handling to ensure all metadata is properly visible

## [1.3.6] - 2024-02-27

Added AI-powered image captioning with Moondream.

### Added
- Automatic image captioning using Moondream AI vision model
- Caption generation for all newly uploaded images
- Display of AI-generated captions on image upload and in database view
- Ability to edit AI captions through the metadata editor

### Changed
- Enhanced metadata structure to include AI captions
- Updated user interface to highlight AI-generated descriptions
- Modified upload form to indicate automatic caption generation
- Improved visual display of captions with styled UI elements

### Technical
- Added moondream package to dependencies
- Implemented caption generation in the image processing pipeline
- Added proper error handling for when API key is not available
- Ensured backwards compatibility with existing images without captions

## [1.3.5] - 2024-02-26

Added multimodal search capability.

### Added
- Combined image and text search functionality
- Weighted multimodal search with adjustable balance between image and text influence
- New homepage section for multimodal search
- Interactive search results with options to adjust weights and search again

### Changed
- Enhanced search interface to present all search options more clearly
- Improved visual design with additional styling for new features

## [1.3.4] - 2024-02-25

Metadata management enhancements for better control over stored images.

### Added
- Edit button for each image in the database view
- Metadata editing form with description and custom metadata fields
- Update functionality for modifying metadata after initial upload
- Improved user experience with clear feedback on metadata updates

### Changed
- Enhanced database view with action buttons for each image
- Updated styles for consistent user interface design

## [1.3.3] - 2024-02-25

UI improvements for better user experience.

### Added
- Duplicate detection notification for uploaded images
- Separate tracking and display for new vs. existing images in sample upload
- Enhanced search interface with clearer distinction between image and text search

### Changed
- Improved homepage layout with more intuitive search options
- Better visual feedback when uploading duplicate images
- More descriptive placeholders and button labels

### Fixed
- Clarified that both text and image search are supported
- Added missing visual feedback for upload status

## [1.3.2] - 2024-02-25

Fixed Pinecone API compatibility issues.

### Changed
- Updated code to support both old and new Pinecone API response formats
- Added compatibility layer to handle changes in Pinecone's response structure
- Made system more resilient to future Pinecone API changes

### Fixed
- Fixed "FetchResponse object has no attribute 'get'" error in image upload
- Resolved compatibility issues with Pinecone API across the application
- Ensured backward compatibility with existing Pinecone indexes and data

## [1.3.1] - 2024-02-25

Enhanced image format support and error handling.

### Added
- Support for AVIF image format via pillow-avif-plugin package
- Improved handling for various image formats (BMP, TIFF, GIF)

### Changed
- Enhanced image opening process with fallback mechanisms
- Better error messages for unsupported formats
- Automatic color mode conversion for non-RGB images
- Corrected package dependencies for AVIF support

### Fixed
- AVIF format upload failures
- Issues with various color modes like RGBA, CMYK

## [1.3.0] - 2024-02-25

Added system reset functionality for easier data management.

### Added
- System reset functionality to clear all stored data
- Confirmation page to prevent accidental data deletion
- Admin section in the user interface
- Links to reset function from both home and image management pages

### Security
- Added confirmation step before critical data deletion operations

## [1.2.0] - 2024-02-25

Major update to add statefulness, de-duplication, and enhanced metadata capabilities.

### Added
- Content-based image hashing to prevent duplicate image storage
- Custom metadata input field in the upload form
- New "View all images" page to browse the image database
- Automatic loading of existing metadata from Pinecone on startup

### Changed
- Switched from random UUIDs to perceptual hashing (imagehash library)
- Enhanced the upload process to check for existing identical images
- Improved the UI with better metadata display and management
- Added deterministic IDs based on image content for better tracking

### Fixed
- Eliminated redundant image processing for duplicate uploads
- Improved metadata persistence through Pinecone integration
- Enhanced error handling for Pinecone connection issues
- Better handling of image metadata throughout the application

### Performance
- Reduced storage requirements by preventing duplicate images
- Improved startup time with better metadata caching
- Enhanced efficiency of the image processing pipeline

## [1.1.0] - 2024-02-24

Major refactoring for improved modularity and performance.

### Added
- Comprehensive logging throughout the application
- Model caching system for improved performance
- Detailed performance measurements for critical operations
- Better error handling with detailed error messages
- Home page enhancement with direct link to upload sample images

### Changed
- Refactored codebase into modular architecture
- Separated model loading and utility functions into utils.py
- Improved dependency management in requirements.txt
- Updated Pinecone library from pinecone-client to pinecone
- Enhanced README with comprehensive documentation
- Added detailed setup and troubleshooting instructions

### Fixed
- Issue with hanging application due to insufficient logging
- Dependency conflicts with outdated pinecone-client package
- Improved error handling for unsupported image formats (e.g., AVIF)
- Added missing environment variables documentation

### Performance
- Reduced model loading time by implementing caching
- Improved image processing with better error handling
- Added timing information for performance-critical operations

### Technical Debt Addressed
- Removed redundant code through modularization
- Improved code organization and readability
- Added type hints for better code quality
- Standardized logging format throughout the application

## Future Development Plans

Features planned for upcoming releases:

1. **User Authentication**
   - User registration and login system
   - Personal image collections

2. **Advanced Search Options**
   - Adjustable similarity thresholds
   - Filter by image attributes
   - Search history
   - Hybrid search combining vector similarity with metadata text search
   - Metadata filtering capabilities based on custom user input
   - Weighted search parameters to balance visual vs. textual relevance

3. **API Improvements**
   - RESTful API for programmatic access
   - Batch processing endpoints
   - Advanced query parameters

4. **Performance Optimizations**
   - Asynchronous processing queue
   - Image preprocessing options
   - Scalability improvements

5. **UI/UX Enhancements**
   - Modern responsive interface
   - Interactive search results
   - Preview thumbnails
   - Drag and drop uploads 