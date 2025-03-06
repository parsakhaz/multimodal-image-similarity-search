'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useUploadStore } from '@/stores/uploadStore';
import { useFilterStore } from '@/stores/filterStore';
import FileDropzone from '@/components/FileDropzone';
import { FiChevronLeft, FiFilter, FiUpload, FiTrash2, FiPlus } from 'react-icons/fi';
import { useEffect } from 'react';

export default function ManagePage() {
  // Upload store
  const {
    file,
    description,
    customMetadata,
    removeBg,
    isUploading,
    error: uploadError,
    success,
    setFile,
    setDescription,
    setCustomMetadata,
    setRemoveBg,
    uploadImage,
    resetForm,
  } = useUploadStore();

  // Filter store
  const {
    filters,
    isLoading: isLoadingFilters,
    error: filterError,
    fetchFilters,
    addFilter,
    deleteFilter,
  } = useFilterStore();

  // Local states
  const [activeTab, setActiveTab] = useState<'upload' | 'filters'>('upload');
  const [newFilter, setNewFilter] = useState('');

  // Fetch filters on component mount
  useEffect(() => {
    fetchFilters();
  }, [fetchFilters]);

  // Handle filter creation
  const handleAddFilter = async () => {
    if (!newFilter.trim()) return;
    
    await addFilter(newFilter);
    setNewFilter('');
  };

  // Handle filter deletion
  const handleDeleteFilter = async (filter: string) => {
    // Optional: add confirmation
    if (window.confirm(`Are you sure you want to delete the filter: ${filter}?`)) {
      await deleteFilter(filter);
    }
  };

  // Show success message after upload
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => {
        resetForm();
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [success, resetForm]);

  return (
    <main className="flex min-h-screen flex-col p-8">
      <div className="max-w-4xl mx-auto w-full">
        {/* Header */}
        <div className="mb-8 flex justify-between items-center">
          <Link href="/" className="flex items-center text-gray-600 hover:text-gray-900">
            <FiChevronLeft className="mr-1" />
            Back to Home
          </Link>
          <h1 className="text-2xl font-bold">Manage Library</h1>
        </div>

        {/* Tabs */}
        <div className="flex border-b mb-6">
          <button
            className={`py-2 px-4 font-medium flex items-center ${
              activeTab === 'upload'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('upload')}
          >
            <FiUpload className="mr-2" />
            Upload Images
          </button>
          
          <button
            className={`py-2 px-4 font-medium flex items-center ${
              activeTab === 'filters'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('filters')}
          >
            <FiFilter className="mr-2" />
            Manage Filters
          </button>
        </div>

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h2 className="text-lg font-medium mb-4">Upload New Image</h2>
            
            {success && (
              <div className="mb-4 p-3 bg-green-50 text-green-700 rounded-md">
                Image uploaded successfully!
              </div>
            )}
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Image
                </label>
                <FileDropzone
                  onFileSelected={setFile}
                  selectedFile={file}
                />
              </div>
              
              <div>
                <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <input
                  type="text"
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe this image"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Provide a detailed description to improve search results
                </p>
              </div>
              
              <div>
                <label htmlFor="metadata" className="block text-sm font-medium text-gray-700 mb-1">
                  Custom Metadata
                </label>
                <textarea
                  id="metadata"
                  value={customMetadata}
                  onChange={(e) => setCustomMetadata(e.target.value)}
                  placeholder="Add any additional metadata (tags, categories, etc.)"
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Optional: Add JSON, key-value pairs, or freeform text
                </p>
              </div>
              
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="removeBg"
                  checked={removeBg}
                  onChange={(e) => setRemoveBg(e.target.checked)}
                  className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                />
                <label htmlFor="removeBg" className="ml-2 text-sm text-gray-700">
                  Remove background (may increase processing time)
                </label>
              </div>
              
              <div>
                <button
                  onClick={uploadImage}
                  disabled={!file || isUploading}
                  className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isUploading ? 'Uploading...' : 'Upload Image'}
                </button>
                
                {uploadError && (
                  <div className="mt-2 text-red-600 text-sm">{uploadError}</div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Filters Tab */}
        {activeTab === 'filters' && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h2 className="text-lg font-medium mb-4">Manage Filters</h2>
            
            <div className="space-y-6">
              {/* Add new filter */}
              <div className="space-y-2">
                <label htmlFor="newFilter" className="block text-sm font-medium text-gray-700">
                  Add New Filter
                </label>
                
                <div className="flex gap-2">
                  <input
                    type="text"
                    id="newFilter"
                    value={newFilter}
                    onChange={(e) => setNewFilter(e.target.value)}
                    placeholder="Enter filter text"
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                  
                  <button
                    onClick={handleAddFilter}
                    disabled={!newFilter.trim() || isLoadingFilters}
                    className="px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <FiPlus className="h-5 w-5" />
                  </button>
                </div>
                
                <p className="text-xs text-gray-500">
                  Filters allow you to organize and quickly find images
                </p>
              </div>
              
              {/* Filter list */}
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">
                  Current Filters {isLoadingFilters && '(Loading...)'}
                </h3>
                
                {filterError && (
                  <div className="mb-2 text-red-600 text-sm">{filterError}</div>
                )}
                
                {filters.length === 0 && !isLoadingFilters ? (
                  <p className="text-sm text-gray-500">No filters defined yet</p>
                ) : (
                  <ul className="space-y-2 max-h-96 overflow-y-auto">
                    {filters.map((filter) => (
                      <li key={filter} className="flex justify-between items-center p-2 bg-gray-50 rounded-md">
                        <span className="text-sm">{filter}</span>
                        
                        <button
                          onClick={() => handleDeleteFilter(filter)}
                          className="text-red-500 hover:text-red-700"
                          aria-label={`Delete filter ${filter}`}
                        >
                          <FiTrash2 className="h-4 w-4" />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
} 