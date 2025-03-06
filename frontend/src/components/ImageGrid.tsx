import React from 'react';
import { ImageMetadata } from '@/types';

interface ImageGridProps {
  images: ImageMetadata[];
  onImageClick?: (image: ImageMetadata) => void;
  className?: string;
  emptyMessage?: string;
}

const ImageGrid: React.FC<ImageGridProps> = ({
  images,
  onImageClick,
  className = '',
  emptyMessage = 'No images found',
}) => {
  if (images.length === 0) {
    return (
      <div className="w-full h-40 flex items-center justify-center">
        <p className="text-gray-500">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <div className={`grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 ${className}`}>
      {images.map((image) => (
        <div
          key={image.id}
          className="relative overflow-hidden rounded-lg shadow-md hover:shadow-lg transition-shadow cursor-pointer"
          onClick={() => onImageClick && onImageClick(image)}
        >
          <div className="aspect-square bg-gray-100 overflow-hidden">
            <img
              src={image.thumbnail_url.startsWith('http') ? image.thumbnail_url : `http://localhost:8000${image.thumbnail_url}`}
              alt={image.description || image.filename}
              className="w-full h-full object-cover transition-transform hover:scale-105"
            />
          </div>
          
          <div className="p-2 bg-white">
            <h3 className="text-sm font-medium truncate" title={image.description || image.filename}>
              {image.description || image.filename}
            </h3>
            
            {image.similarity_score !== undefined && (
              <p className="text-xs text-gray-500">
                Match: {Math.round(image.similarity_score * 100)}%
              </p>
            )}
          </div>
          
          {image.custom_metadata && (
            <div className="absolute top-2 right-2 bg-blue-100 text-blue-800 text-xs px-1 rounded">
              <span className="sr-only">Has metadata</span>
              <span>✓</span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default ImageGrid; 