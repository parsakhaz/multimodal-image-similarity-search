import React from 'react';
import { ImageMetadata } from '@/types';

interface ImageDetailModalProps {
  image: ImageMetadata;
  onClose: () => void;
}

const ImageDetailModal: React.FC<ImageDetailModalProps> = ({ image, onClose }) => {
  if (!image) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
      {/* Modal container - 80% of screen size */}
      <div className="bg-white rounded-lg w-[80%] h-[80%] max-h-[80vh] overflow-hidden flex flex-col">
        <div className="p-4 border-b flex justify-between items-center">
          <h3 className="text-lg font-medium">Image Details</h3>
          <button 
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            ✕
          </button>
        </div>
        
        <div className="flex flex-col md:flex-row overflow-auto p-4 gap-6 h-full">
          {/* Image container - left side */}
          <div className="md:w-3/5 h-full flex items-center justify-center">
            <img 
              src={image.url?.startsWith('http') ? image.url : `http://localhost:8000${image.url}`} 
              alt={image.description || image.filename}
              className="max-w-full max-h-full object-contain"
            />
          </div>
          
          {/* Metadata container - right side */}
          <div className="md:w-2/5 space-y-4">
            <div>
              <h4 className="text-sm font-medium text-gray-500">Filename</h4>
              <p className="text-lg">{image.filename}</p>
            </div>
            
            {image.description && (
              <div>
                <h4 className="text-sm font-medium text-gray-500">Description</h4>
                <p className="text-lg">{image.description}</p>
              </div>
            )}
            
            {image.custom_metadata && (
              <div>
                <h4 className="text-sm font-medium text-gray-500">Custom Metadata</h4>
                <p className="whitespace-pre-wrap">{image.custom_metadata}</p>
              </div>
            )}
            
            {image.similarity_score !== undefined && (
              <div>
                <h4 className="text-sm font-medium text-gray-500">Similarity Score</h4>
                <div className="mt-1 relative pt-1">
                  <div className="overflow-hidden h-4 mb-1 text-xs flex rounded bg-blue-100">
                    <div 
                      style={{ width: `${Math.round(image.similarity_score * 100)}%` }} 
                      className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"
                    >
                      {Math.round(image.similarity_score * 100)}%
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div>
              <h4 className="text-sm font-medium text-gray-500">Added On</h4>
              <p>{new Date(image.created_at).toLocaleDateString()}</p>
            </div>
            
            <div className="mt-4">
              <button 
                onClick={() => window.open(image.url?.startsWith('http') ? image.url : `http://localhost:8000${image.url}`, '_blank')}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                View Full Image
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageDetailModal; 