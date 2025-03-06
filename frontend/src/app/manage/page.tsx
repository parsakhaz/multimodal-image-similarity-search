'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useUploadStore } from '@/stores/uploadStore';
import { useFilterStore } from '@/stores/filterStore';
import FileDropzone from '@/components/FileDropzone';
import { FiChevronLeft, FiFilter, FiUpload, FiTrash2, FiPlus, FiAlertTriangle } from 'react-icons/fi';
import { useEffect } from 'react';
import apiClient from '@/lib/api';
import { useRouter } from 'next/navigation';

// Define proper type for filter progress data
interface FilterProgressData {
  status: string;
  progress: number;
  error?: string;
  processed?: number;
  total?: number;
  current_image?: string;
  message?: string;
}

export default function ManagePage() {
  const router = useRouter();
  
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
  const [activeTab, setActiveTab] = useState<'upload' | 'filters' | 'batch'>('upload');
  const [newFilter, setNewFilter] = useState('');
  const [isResetting, setIsResetting] = useState(false);
  const [resetConfirm, setResetConfirm] = useState(false);
  const [filterProgress, setFilterProgress] = useState<Record<string, FilterProgressData>>({});
  
  // Folder upload states
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [folderRemoveBg, setFolderRemoveBg] = useState(false);
  const [isUploadingFolder, setIsUploadingFolder] = useState(false);
  const [folderUploadResults, setFolderUploadResults] = useState<{
    success: boolean;
    total: number;
    successful: number;
    skipped: number;
    failed: number;
    results: Array<{
      filename: string;
      status: string;
      id?: string;
      reason?: string;
    }>;
  } | null>(null);
  const [folderUploadError, setFolderUploadError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<{
    status: string;
    current: number;
    total: number;
    percent: number;
    filterInfo?: {
      currentFilter?: string;
      filterIndex?: number;
      totalFilters?: number;
    }
  } | null>(null);

  // Fetch filters on component mount
  useEffect(() => {
    fetchFilters();
  }, [fetchFilters]);

  // Handle filter creation
  const handleAddFilter = async () => {
    if (!newFilter.trim()) return;
    
    try {
      // Add the filter
      await addFilter(newFilter);
      setNewFilter('');
      
      // Initialize filter progress tracking with processing state
      setFilterProgress({
        ...filterProgress,
        [newFilter]: { status: 'initializing', progress: 0 }
      });
      
      // Start tracking the progress with a more accurate polling mechanism
      let processComplete = false;
      let pollingCounter = 0;
      
      const checkProgress = setInterval(async () => {
        try {
          const response = await apiClient.getFilterProgress(newFilter);
          const progressData = response.data;
          
          // Only update if we got valid data
          if (progressData && typeof progressData === 'object') {
            // Check if we have a valid status
            if (progressData.status === 'processing' || progressData.status === 'completed') {
              // Direct mapping of backend progress data to our state
              setFilterProgress(prev => ({
                ...prev,
                [newFilter]: progressData
              }));
              
              // If completed, start a timer to clean up the progress display
              if (progressData.status === 'completed') {
                processComplete = true;
                clearInterval(checkProgress);
                
                // Refresh filter list after completion
                setTimeout(() => {
                  fetchFilters();
                  setFilterProgress(prev => {
                    const updated = { ...prev };
                    delete updated[newFilter];
                    return updated;
                  });
                }, 2000);
              }
            }
          } else if (pollingCounter < 10) {
            // It might take a moment for the backend to create the progress entry
            // Keep polling for a few seconds before showing not_found
            pollingCounter++;
            
            // Let the user know we're waiting for the backend to start processing
            if (pollingCounter > 2) {
              setFilterProgress(prev => ({
                ...prev,
                [newFilter]: {
                  status: 'not_found',
                  progress: 0,
                  message: 'Waiting for processing to begin...'
                }
              }));
            }
          } else if (!processComplete) {
            // Only update to not_found if we haven't completed
            setFilterProgress(prev => ({
              ...prev,
              [newFilter]: {
                status: 'not_found',
                progress: 0
              }
            }));
          }
        } catch (error) {
          console.error('Error checking filter progress:', error);
          if (!processComplete) {
            setFilterProgress(prev => ({
              ...prev,
              [newFilter]: {
                status: 'error',
                progress: 0,
                error: 'Failed to check progress'
              }
            }));
          }
        }
      }, 500); // Poll more frequently for smoother updates
      
      // Clear interval after 5 minutes to prevent endless polling
      setTimeout(() => {
        if (!processComplete) {
          clearInterval(checkProgress);
          setFilterProgress(prev => {
            const updated = { ...prev };
            delete updated[newFilter];
            return updated;
          });
        }
      }, 300000);
      
    } catch (error) {
      console.error('Error adding filter:', error);
    }
  };

  // Handle filter deletion
  const handleDeleteFilter = async (filter: string) => {
    // Optional: add confirmation
    if (window.confirm(`Are you sure you want to delete the filter: ${filter}?`)) {
      await deleteFilter(filter);
    }
  };
  
  // Handle system reset
  const handleResetSystem = async () => {
    if (!resetConfirm) {
      setResetConfirm(true);
      return;
    }
    
    try {
      setIsResetting(true);
      await apiClient.resetSystem();
      alert('System reset successful! All images and filters have been deleted.');
      setResetConfirm(false);
      // Refresh the page to show empty state
      router.refresh();
      
    } catch (error) {
      console.error('Error resetting system:', error);
      alert('Failed to reset system. Please try again.');
    } finally {
      setIsResetting(false);
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

  // Handle folder upload
  const handleFolderUpload = async () => {
    if (folderFiles.length === 0) {
      setFolderUploadError('Please select files to upload');
      return;
    }
    
    try {
      setIsUploadingFolder(true);
      setFolderUploadError(null);
      setUploadProgress({
        status: 'Starting upload...',
        current: 0,
        total: folderFiles.length,
        percent: 0
      });
      
      // Use the progress-tracking version of the upload
      const response = await apiClient.uploadFolderWithProgress({
        files: folderFiles,
        removeBg: folderRemoveBg,
        onProgress: (status, current, total, filterInfo) => {
          setUploadProgress({
            status,
            current,
            total,
            percent: Math.round((current / total) * 100),
            filterInfo
          });
        }
      });
      
      setFolderUploadResults(response.data);
      // Clear the files after successful upload
      setFolderFiles([]);
      
    } catch (error) {
      console.error('Error uploading folder:', error);
      setFolderUploadError('An error occurred while uploading the files. Please try again.');
    } finally {
      setIsUploadingFolder(false);
      // Reset upload progress immediately instead of using a timeout
      setUploadProgress(null);
    }
  };
  
  // Handle file selection for folder upload
  const handleFolderFilesSelected = (files: FileList | null) => {
    if (!files) return;
    setFolderFiles(Array.from(files));
    setFolderUploadResults(null);
    setFolderUploadError(null);
  };

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
            Upload Image
          </button>
          
          <button
            className={`py-2 px-4 font-medium flex items-center ${
              activeTab === 'batch'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('batch')}
          >
            <FiUpload className="mr-2" />
            Batch Upload
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
        
        {/* Batch Upload Tab */}
        {activeTab === 'batch' && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h2 className="text-lg font-medium mb-4">Batch Upload Images</h2>
            
            {folderUploadResults && (
              <div className="mb-4 p-3 bg-green-50 text-green-700 rounded-md">
                <p className="font-medium">Upload complete!</p>
                <p>Successfully uploaded: {folderUploadResults.successful} images</p>
                <p>Skipped (duplicates): {folderUploadResults.skipped} images</p>
                <p>Failed: {folderUploadResults.failed} images</p>
              </div>
            )}
            
            {folderUploadError && (
              <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-md">
                {folderUploadError}
              </div>
            )}
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Select Multiple Images
                </label>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-blue-400 hover:bg-gray-50 transition-colors">
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={(e) => handleFolderFilesSelected(e.target.files)}
                    className="w-full"
                  />
                  <p className="mt-2 text-xs text-gray-500 text-center">
                    Select multiple image files to upload them in batch
                  </p>
                </div>
                
                {folderFiles.length > 0 && (
                  <p className="mt-2 text-sm text-gray-600">
                    {folderFiles.length} file(s) selected
                  </p>
                )}
              </div>
              
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="folderRemoveBg"
                  checked={folderRemoveBg}
                  onChange={(e) => setFolderRemoveBg(e.target.checked)}
                  className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                />
                <label htmlFor="folderRemoveBg" className="ml-2 text-sm text-gray-700">
                  Remove background from all images (increases processing time)
                </label>
              </div>
              
              <div>
                <button
                  onClick={handleFolderUpload}
                  disabled={folderFiles.length === 0 || isUploadingFolder}
                  className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isUploadingFolder ? 'Uploading...' : `Upload ${folderFiles.length} File(s)`}
                </button>
              </div>
              
              {/* Progress display - only show when actively uploading */}
              {isUploadingFolder && uploadProgress && (
                <div className="mt-4 p-4 bg-blue-50 rounded-md">
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium text-blue-700">{uploadProgress.status}</span>
                    <span className="text-sm text-blue-700">{uploadProgress.percent}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
                      style={{ width: `${uploadProgress.percent}%` }}
                    ></div>
                  </div>
                  <div className="text-xs text-blue-700 mt-1 text-right">
                    {uploadProgress.current} of {uploadProgress.total} files
                  </div>
                  
                  {/* Filter progress */}
                  {uploadProgress.filterInfo && 
                   uploadProgress.filterInfo.totalFilters && 
                   uploadProgress.filterInfo.filterIndex && 
                   uploadProgress.filterInfo.totalFilters > 0 && (
                    <div className="mt-3 border-t border-blue-200 pt-2">
                      <div className="flex justify-between mb-1">
                        <span className="text-xs font-medium text-blue-700">Filter Progress:</span>
                        <span className="text-xs text-blue-700">
                          {uploadProgress.filterInfo.filterIndex} of {uploadProgress.filterInfo.totalFilters}
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5 mb-1">
                        <div 
                          className="bg-green-500 h-1.5 rounded-full transition-all duration-300" 
                          style={{ 
                            width: `${Math.round(((uploadProgress.filterInfo.filterIndex || 0) / (uploadProgress.filterInfo.totalFilters || 1)) * 100)}%` 
                          }}
                        ></div>
                      </div>
                      <div className="text-xs text-blue-700 mt-1">
                        <span className="font-medium">Current filter:</span> {uploadProgress.filterInfo.currentFilter || 'Unknown'}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* Results display */}
              {folderUploadResults && folderUploadResults.results && folderUploadResults.results.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium mb-2">Upload Results:</h3>
                  <div className="max-h-60 overflow-y-auto border rounded">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Filename</th>
                          <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {folderUploadResults.results.map((result, index) => (
                          <tr key={index} className={
                            result.status === 'success' ? 'bg-green-50' :
                            result.status === 'skipped' ? 'bg-yellow-50' :
                            'bg-red-50'
                          }>
                            <td className="px-4 py-2 text-sm">{result.filename}</td>
                            <td className="px-4 py-2 text-sm">
                              {result.status === 'success' && <span className="text-green-600">Success</span>}
                              {result.status === 'skipped' && (
                                <span className="text-amber-600">
                                  Skipped
                                  {result.reason && 
                                    <span className="text-xs ml-2">
                                      ({result.reason})
                                      {result.id && <span className="italic ml-1">ID: {result.id.substring(0, 8)}...</span>}
                                    </span>
                                  }
                                </span>
                              )}
                              {result.status === 'error' && (
                                <span className="text-red-600">
                                  Error
                                  {result.reason && <span className="text-xs ml-2">({result.reason})</span>}
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
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
                
                {/* Filter Progress Section */}
                {filterProgress && Object.keys(filterProgress).length > 0 && (
                  <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                    <h3 className="text-lg font-medium text-blue-900 mb-2">Filter Processing</h3>
                    {Object.entries(filterProgress).map(([filter, data]) => (
                      <div key={filter} className="mb-3">
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium text-blue-900">{filter}</span>
                          <span className="text-sm text-blue-700">
                            {data.status === 'processing' ? `${Math.round(data.progress)}%` : 
                             data.status === 'completed' ? 'Complete' : 
                             data.status === 'not_found' ? 'Waiting to start...' : 
                             data.status === 'error' ? 'Error' : 
                             data.status === 'initializing' ? 'Starting...' : 'Unknown'}
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                          <div 
                            className={`h-2 rounded-full ${
                              data.status === 'completed' ? 'bg-green-500' : 
                              data.status === 'error' ? 'bg-red-500' : 
                              'bg-blue-500'
                            }`}
                            style={{ width: `${data.progress}%` }}
                          ></div>
                        </div>
                        {data.status === 'processing' && data.processed !== undefined && data.total !== undefined && (
                          <p className="text-xs text-blue-700">
                            Processing image {data.processed}/{data.total}
                            {data.current_image && (
                              <span className="ml-1">({data.current_image.split('_').pop()?.substring(0, 8)})</span>
                            )}
                          </p>
                        )}
                        {data.status === 'error' && data.error && (
                          <p className="text-xs text-red-500">{data.error}</p>
                        )}
                        {data.status === 'not_found' && (
                          <p className="text-xs text-blue-500">
                            {data.message || "Preparing to process images..."}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
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
        
        {/* Reset System Button */}
        <div className="mt-8 border-t pt-6">
          <div className="bg-red-50 rounded-lg p-4 border border-red-200">
            <h3 className="text-lg font-medium text-red-700 flex items-center">
              <FiAlertTriangle className="mr-2" />
              Danger Zone
            </h3>
            <p className="mt-1 text-sm text-red-600">
              This action will permanently delete all images and filters from the system.
            </p>
            
            <div className="mt-4">
              {resetConfirm ? (
                <div className="space-y-3">
                  <p className="text-sm font-medium text-red-700">Are you sure you want to delete everything?</p>
                  <div className="flex space-x-3">
                    <button
                      onClick={handleResetSystem}
                      disabled={isResetting}
                      className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50"
                    >
                      {isResetting ? 'Deleting...' : 'Yes, Delete Everything'}
                    </button>
                    <button
                      onClick={() => setResetConfirm(false)}
                      className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  onClick={handleResetSystem}
                  className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                >
                  Reset System / Delete All Data
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
} 