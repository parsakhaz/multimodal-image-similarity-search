import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FiUpload } from 'react-icons/fi';

interface FileDropzoneProps {
  onFileSelected: (file: File) => void;
  accept?: Record<string, string[]>;
  maxSize?: number;
  className?: string;
  label?: string;
  showPreview?: boolean;
  selectedFile?: File | null;
}

const FileDropzone: React.FC<FileDropzoneProps> = ({
  onFileSelected,
  accept = {
    'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp', '.avif']
  },
  maxSize = 10 * 1024 * 1024, // 10MB
  className = '',
  label = 'Drag & drop an image here, or click to select',
  showPreview = true,
  selectedFile = null,
}) => {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelected(acceptedFiles[0]);
      }
    },
    [onFileSelected]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    maxSize,
    multiple: false,
  });

  // Preview URL for the selected file
  const previewUrl = selectedFile ? URL.createObjectURL(selectedFile) : null;

  // Clean up the preview URL when component unmounts or file changes
  React.useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <div className={`w-full ${className}`}>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer transition-colors
          ${
            isDragActive
              ? 'border-blue-500 bg-blue-50'
              : isDragReject
              ? 'border-red-500 bg-red-50'
              : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }
        `}
      >
        <input {...getInputProps()} />
        
        {showPreview && previewUrl ? (
          <div className="flex flex-col items-center space-y-4">
            <img
              src={previewUrl}
              alt="Preview"
              className="max-h-40 max-w-full rounded-md object-contain"
            />
            <p className="text-sm text-gray-500">Click or drag to replace</p>
          </div>
        ) : (
          <div className="flex flex-col items-center space-y-4">
            <FiUpload className="h-10 w-10 text-gray-400" />
            <p className="text-sm text-center text-gray-500">{label}</p>
            <p className="text-xs text-gray-400">
              (Max file size: {Math.floor(maxSize / (1024 * 1024))}MB)
            </p>
          </div>
        )}
      </div>
      
      {isDragReject && (
        <p className="mt-2 text-sm text-red-500">
          File type not accepted or too large
        </p>
      )}
      
      {selectedFile && !showPreview && (
        <p className="mt-2 text-sm text-gray-500">
          Selected: {selectedFile.name}
        </p>
      )}
    </div>
  );
};

export default FileDropzone; 