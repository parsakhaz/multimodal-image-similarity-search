'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useSearchStore } from '@/stores/searchStore';
import ImageGrid from '@/components/ImageGrid';
import FileDropzone from '@/components/FileDropzone';
import FilterSelector from '@/components/FilterSelector';
import { FiChevronLeft, FiSearch, FiImage, FiLayers } from 'react-icons/fi';
import { ImageMetadata } from '@/types';

export default function SearchPage() {
  const {
    searchType,
    query,
    selectedImage,
    weightImage,
    filters,
    results,
    isLoading,
    error,
    setSearchType,
    setQuery,
    setSelectedImage,
    setWeightImage,
    setFilters,
    searchByText,
    searchByImage,
    searchMultimodal,
  } = useSearchStore();

  const [selectedResult, setSelectedResult] = useState<ImageMetadata | null>(null);

  // Reset selected result when results change
  useEffect(() => {
    setSelectedResult(null);
  }, [results]);

  const handleSearch = async () => {
    if (searchType === 'text') {
      await searchByText();
    } else if (searchType === 'image') {
      await searchByImage();
    } else if (searchType === 'multimodal') {
      await searchMultimodal();
    }
  };

  const handleImageClick = (image: ImageMetadata) => {
    setSelectedResult(image);
  };

  return (
    <main className="flex min-h-screen flex-col p-8">
      <div className="max-w-7xl mx-auto w-full">
        {/* Header */}
        <div className="mb-8 flex justify-between items-center">
          <Link href="/" className="flex items-center text-gray-600 hover:text-gray-900">
            <FiChevronLeft className="mr-1" />
            Back to Home
          </Link>
          <h1 className="text-2xl font-bold">Image Search</h1>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Search Panel */}
          <div className="lg:col-span-1 bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <div className="space-y-6">
              {/* Search Type Selector */}
              <div>
                <h2 className="text-lg font-medium mb-3">Search Method</h2>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    className={`py-2 px-3 text-sm rounded-md flex items-center justify-center ${
                      searchType === 'text'
                        ? 'bg-blue-100 text-blue-700 font-medium'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    onClick={() => setSearchType('text')}
                  >
                    <FiSearch className="mr-1" />
                    Text
                  </button>
                  <button
                    className={`py-2 px-3 text-sm rounded-md flex items-center justify-center ${
                      searchType === 'image'
                        ? 'bg-blue-100 text-blue-700 font-medium'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    onClick={() => setSearchType('image')}
                  >
                    <FiImage className="mr-1" />
                    Image
                  </button>
                  <button
                    className={`py-2 px-3 text-sm rounded-md flex items-center justify-center ${
                      searchType === 'multimodal'
                        ? 'bg-blue-100 text-blue-700 font-medium'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    onClick={() => setSearchType('multimodal')}
                  >
                    <FiLayers className="mr-1" />
                    Both
                  </button>
                </div>
              </div>

              {/* Search Inputs */}
              <div className="space-y-4">
                {(searchType === 'text' || searchType === 'multimodal') && (
                  <div>
                    <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-1">
                      Text Query
                    </label>
                    <input
                      type="text"
                      id="query"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Describe what you're looking for..."
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                )}

                {(searchType === 'image' || searchType === 'multimodal') && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Image Query
                    </label>
                    <FileDropzone
                      onFileSelected={setSelectedImage}
                      selectedFile={selectedImage}
                      label="Drop an image to search by similarity"
                    />
                  </div>
                )}

                {searchType === 'multimodal' && (
                  <div>
                    <label htmlFor="weight" className="block text-sm font-medium text-gray-700 mb-1">
                      Image Weight: {weightImage.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      id="weight"
                      min="0"
                      max="1"
                      step="0.1"
                      value={weightImage}
                      onChange={(e) => setWeightImage(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Text</span>
                      <span>Equal</span>
                      <span>Image</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Filters */}
              <FilterSelector
                selectedFilters={filters}
                onFilterChange={setFilters}
              />

              {/* Search Button */}
              <button
                onClick={handleSearch}
                disabled={isLoading}
                className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Searching...' : 'Search'}
              </button>

              {error && (
                <div className="mt-2 text-red-600 text-sm">{error}</div>
              )}
            </div>
          </div>

          {/* Results Area */}
          <div className="lg:col-span-3">
            {results.length > 0 ? (
              <div>
                <h2 className="text-xl font-medium mb-4">Search Results</h2>
                <ImageGrid
                  images={results}
                  onImageClick={handleImageClick}
                />
              </div>
            ) : (
              <div className="bg-gray-50 rounded-xl p-10 text-center">
                <FiSearch className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h2 className="text-xl font-medium text-gray-600 mb-2">
                  No results to display
                </h2>
                <p className="text-gray-500 max-w-md mx-auto">
                  Use the search panel on the left to find images by text description, 
                  visual similarity, or a combination of both.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Selected Image Modal */}
        {selectedResult && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
              <div className="p-4 border-b flex justify-between items-center">
                <h3 className="text-lg font-medium">Image Details</h3>
                <button 
                  onClick={() => setSelectedResult(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
              
              <div className="flex flex-col md:flex-row overflow-auto p-4 gap-6">
                <div className="md:w-1/2">
                  <img 
                    src={selectedResult.url} 
                    alt={selectedResult.description || selectedResult.filename}
                    className="w-full h-auto object-contain max-h-[60vh]"
                  />
                </div>
                
                <div className="md:w-1/2 space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-500">Filename</h4>
                    <p>{selectedResult.filename}</p>
                  </div>
                  
                  {selectedResult.description && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-500">Description</h4>
                      <p>{selectedResult.description}</p>
                    </div>
                  )}
                  
                  {selectedResult.custom_metadata && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-500">Custom Metadata</h4>
                      <p className="whitespace-pre-wrap">{selectedResult.custom_metadata}</p>
                    </div>
                  )}
                  
                  {selectedResult.similarity_score !== undefined && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-500">Similarity Score</h4>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full" 
                          style={{ width: `${Math.round(selectedResult.similarity_score * 100)}%` }}
                        ></div>
                      </div>
                      <p className="text-sm text-right mt-1">
                        {Math.round(selectedResult.similarity_score * 100)}%
                      </p>
                    </div>
                  )}
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-500">Added On</h4>
                    <p>{new Date(selectedResult.created_at).toLocaleDateString()}</p>
                  </div>
                </div>
              </div>
              
              <div className="p-4 border-t flex justify-end">
                <Link
                  href={`/images?id=${selectedResult.id}`}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  View in Collection
                </Link>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
} 